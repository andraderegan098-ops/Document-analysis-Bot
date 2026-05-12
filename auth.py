"""
auth.py — DocuBot Authentication System
----------------------------------------
- Hashed passwords (bcrypt)
- User stored in .streamlit/users.json
- Session-based login/logout
- Admin can add/remove users
- First-run creates a default admin account
"""

import os
import json
import hashlib
import hmac
import streamlit as st
from datetime import datetime

USERS_FILE = os.path.join(".streamlit", "users.json")


# ── Password hashing (no bcrypt dependency — uses PBKDF2) ─────────────────────

def _hash_password(password: str) -> str:
    """Return a secure PBKDF2-SHA256 hash of the password."""
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return salt.hex() + ":" + key.hex()


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2 hash."""
    try:
        salt_hex, key_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
        return hmac.compare_digest(new_key, key)
    except Exception:
        return False


# ── User store ────────────────────────────────────────────────────────────────

def _load_users() -> dict:
    """Load users from JSON file. Returns empty dict if file missing (first run)."""
    os.makedirs(".streamlit", exist_ok=True)
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def is_first_run() -> bool:
    """True if no users exist yet — app needs initial admin setup."""
    return len(_load_users()) == 0


def _save_users(users: dict):
    os.makedirs(".streamlit", exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def add_user(username: str, password: str, name: str, role: str = "user") -> bool:
    """Add a new user. Returns False if username already exists."""
    users = _load_users()
    if username in users:
        return False
    users[username] = {
        "password_hash": _hash_password(password),
        "role": role,
        "name": name,
        "created_at": datetime.now().isoformat(),
    }
    _save_users(users)
    return True


def delete_user(username: str) -> bool:
    """Delete a user. Returns False if user doesn't exist or is last admin."""
    users = _load_users()
    if username not in users:
        return False
    # Don't delete last admin
    admins = [u for u, d in users.items() if d["role"] == "admin"]
    if username in admins and len(admins) == 1:
        return False
    del users[username]
    _save_users(users)
    return True


def change_password(username: str, new_password: str) -> bool:
    users = _load_users()
    if username not in users:
        return False
    users[username]["password_hash"] = _hash_password(new_password)
    _save_users(users)
    return True


def authenticate(username: str, password: str):
    """
    Verify credentials. Returns user dict on success, None on failure.
    """
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    if not _verify_password(password, user["password_hash"]):
        return None
    return {"username": username, "name": user["name"], "role": user["role"]}


def get_all_users() -> dict:
    return _load_users()


# ── Streamlit UI helpers ──────────────────────────────────────────────────────

def is_logged_in() -> bool:
    return st.session_state.get("auth_user") is not None


def current_user() -> dict | None:
    return st.session_state.get("auth_user")


def logout():
    """Clear auth session state."""
    for key in ["auth_user", "messages", "memory", "last_audio",
                 "detected_lang", "pending_image"]:
        st.session_state.pop(key, None)


def render_login_page():
    """
    Render the full-page login UI.
    On first run (no users exist), shows account creation form instead.
    Returns True if login succeeded (caller should st.rerun()).
    """
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 2rem 0 1rem 0;'>
            <span style='font-size:3.5rem;'>🤖</span>
            <h1 style='margin:0.2rem 0 0 0; font-size:2rem;'>DocuBot</h1>
            <p style='color:grey; margin:0;'>Intelligent Document Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── First run: no users exist yet ────────────────────────────────────
        if is_first_run():
            st.subheader("🛠️ Initial Setup")
            st.info("No accounts found. Create your admin account to get started.")

            with st.form("setup_form", clear_on_submit=False):
                setup_name     = st.text_input("Your Full Name", placeholder="e.g. Jane Smith")
                setup_username = st.text_input("Choose a Username", placeholder="e.g. jane")
                setup_password = st.text_input("Choose a Password", type="password")
                setup_confirm  = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Admin Account", use_container_width=True)

                if submitted:
                    if not all([setup_name, setup_username, setup_password, setup_confirm]):
                        st.error("All fields are required.")
                    elif len(setup_password) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif setup_password != setup_confirm:
                        st.error("Passwords do not match.")
                    else:
                        add_user(setup_username.strip(), setup_password,
                                 setup_name.strip(), role="admin")
                        user = authenticate(setup_username.strip(), setup_password)
                        st.session_state["auth_user"] = user
                        st.success(f"✅ Admin account created! Welcome, {setup_name}!")
                        return True
            return False

        # ── Normal login ──────────────────────────────────────────────────────
        st.subheader("🔐 Sign In")

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = authenticate(username.strip(), password)
                    if user:
                        st.session_state["auth_user"] = user
                        st.success(f"Welcome back, {user['name']}! 👋")
                        return True
                    else:
                        st.error("❌ Invalid username or password.")

    return False


def render_user_management():
    """Admin panel to manage users — call inside the sidebar or a page."""
    st.subheader("👥 User Management")
    users = get_all_users()

    # List users
    for uname, udata in users.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        col1.write(f"**{udata['name']}** (`{uname}`)")
        col2.write(f"🏷️ {udata['role']}")
        if uname != current_user()["username"]:  # can't delete yourself
            if col3.button("🗑️", key=f"del_{uname}", help=f"Delete {uname}"):
                if delete_user(uname):
                    st.success(f"Deleted user: {uname}")
                    st.rerun()
                else:
                    st.error("Cannot delete the last admin.")

    st.markdown("---")

    # Add user form
    with st.expander("➕ Add New User"):
        with st.form("add_user_form"):
            new_name = st.text_input("Full Name")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"])
            if st.form_submit_button("Add User"):
                if not all([new_name, new_username, new_password]):
                    st.error("All fields are required.")
                elif add_user(new_username.strip(), new_password, new_name.strip(), new_role):
                    st.success(f"✅ User '{new_username}' created successfully!")
                    st.rerun()
                else:
                    st.error(f"Username '{new_username}' already exists.")

    # Change password form
    with st.expander("🔑 Change Password"):
        with st.form("change_pw_form"):
            target_user = st.selectbox("User", list(users.keys()))
            new_pw = st.text_input("New Password", type="password")
            confirm_pw = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Update Password"):
                if not new_pw:
                    st.error("Password cannot be empty.")
                elif new_pw != confirm_pw:
                    st.error("Passwords do not match.")
                else:
                    change_password(target_user, new_pw)
                    st.success(f"✅ Password updated for '{target_user}'.")
