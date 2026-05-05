"""
Configuration for Zoho Face Recognition Module.
All values are loaded from environment variables.
Update your Render environment variables to match your Zoho Creator setup.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Zoho OAuth Credentials ───────────────────────────────────────────────────
ZOHO_CLIENT_ID = os.environ.get("ZOHO_CLIENT_ID", "")
ZOHO_CLIENT_SECRET = os.environ.get("ZOHO_CLIENT_SECRET", "")
ZOHO_REFRESH_TOKEN = os.environ.get("ZOHO_REFRESH_TOKEN", "")

# ─── Zoho Creator App Config ──────────────────────────────────────────────────
ZOHO_ACCOUNT_OWNER = os.environ.get("ZOHO_ACCOUNT_OWNER", "admin_trrainfoundation")
ZOHO_APP_NAME      = os.environ.get("ZOHO_APP_NAME", "trrain")
ZOHO_DATA_CENTER   = os.environ.get("ZOHO_DATA_CENTER", "in")

# Report / form link names
ZOHO_STUDENT_REPORT   = os.environ.get("ZOHO_STUDENT_REPORT",   "All_Trainees")
ZOHO_ATTENDANCE_FORM  = os.environ.get("ZOHO_ATTENDANCE_FORM",  "Face_Attendance")
ZOHO_ATTENDANCE_REPORT = os.environ.get("ZOHO_ATTENDANCE_REPORT", "All_Face_Attendances")

# ─── Student Database field names ─────────────────────────────────────────────
FIELD_STUDENT_ID        = "ID"   # Zoho system record ID — always present
FIELD_STUDENT_NUMBER    = os.environ.get("FIELD_STUDENT_NUMBER",    "Registration_No")
FIELD_STUDENT_NAME      = os.environ.get("FIELD_STUDENT_NAME",      "Name")
FIELD_STUDENT_PHOTO     = os.environ.get("FIELD_STUDENT_PHOTO",     "Upload_Photo1")

# Multi-line text field to cache the pre-computed 512-d ArcFace embedding (JSON list)
# Add this field in Zoho Creator: Student Database → Multi Line field → link name: Face_Embedding
FIELD_STUDENT_EMBEDDING = os.environ.get("FIELD_STUDENT_EMBEDDING", "Face_Embedding")

# ─── Attendance form field names ──────────────────────────────────────────────
FIELD_ATT_STUDENT = os.environ.get("FIELD_ATT_STUDENT", "Trainee")   # lookup
FIELD_ATT_DATE    = os.environ.get("FIELD_ATT_DATE",    "Date_field")
FIELD_ATT_STATUS  = os.environ.get("FIELD_ATT_STATUS",  "Attendance")   # dropdown

# ─── Face Recognition Settings ────────────────────────────────────────────────
FACE_MATCH_TOLERANCE = float(os.environ.get("FACE_MATCH_TOLERANCE", "0.40"))
CACHE_TTL_SECONDS    = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

# ─── User Management (for center-scoped student lookup) ───────────────────────
# Report link name of the User Management form/report in Zoho Creator
ZOHO_USER_MGMT_REPORT = os.environ.get("ZOHO_USER_MGMT_REPORT", "All_Users")
# Field link name of the email field in the user management form
FIELD_USER_EMAIL   = os.environ.get("FIELD_USER_EMAIL",   "Zoho_ID")
# Field link name of the multiselect Centers lookup in user management
FIELD_USER_CENTERS = os.environ.get("FIELD_USER_CENTERS", "Centre_List")
# Field link name of the Center lookup in the student database
FIELD_STUDENT_CENTER = os.environ.get("FIELD_STUDENT_CENTER", "Centre_Name")

# ─── App Settings ─────────────────────────────────────────────────────────────
PORT       = int(os.environ.get("PORT", 5000))
DEBUG      = os.environ.get("DEBUG", "false").lower() == "true"
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key-in-production")

# Self URL for the always-on keepalive ping (set to your Render URL)
# e.g. https://face-attendance-3wel.onrender.com
SELF_URL = os.environ.get("SELF_URL", "")

# ─── Render API (for auto-updating ZOHO_REFRESH_TOKEN) ────────────────────────
# Get your API key: Render dashboard → Account Settings → API Keys → Create API Key
# Get your Service ID: Render dashboard → your service → Settings → Service ID (srv-xxxxx)
RENDER_API_KEY    = os.environ.get("RENDER_API_KEY", "")
RENDER_SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "")

# Secret passcode to protect the /admin/reauth page from public access
# Set this to any random string in your Render environment variables
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "train-admin-2026")
