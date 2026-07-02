"""
Configuration for Zoho Face Recognition Module.
All values are loaded from environment variables.
Update your Render environment variables to match your Zoho Creator setup.
"""

import logging
import os
from dotenv import load_dotenv

load_dotenv()

_cfg_logger = logging.getLogger(__name__)

# ─── Zoho OAuth Credentials ───────────────────────────────────────────────────
ZOHO_CLIENT_ID = os.environ.get("ZOHO_CLIENT_ID", "")
ZOHO_CLIENT_SECRET = os.environ.get("ZOHO_CLIENT_SECRET", "")
ZOHO_REFRESH_TOKEN = os.environ.get("ZOHO_REFRESH_TOKEN", "")

# ─── Zoho Creator App Config ──────────────────────────────────────────────────
ZOHO_ACCOUNT_OWNER = os.environ.get("ZOHO_ACCOUNT_OWNER", "")
if not ZOHO_ACCOUNT_OWNER:
    # Hardcoded fallback so existing deployments don't break, but log loudly
    ZOHO_ACCOUNT_OWNER = "admin_trrainfoundation"
    _cfg_logger.critical(
        "ZOHO_ACCOUNT_OWNER env var is not set — falling back to hardcoded 'admin_trrainfoundation'. "
        "Set ZOHO_ACCOUNT_OWNER in Render environment variables to remove this risk."
    )
ZOHO_APP_NAME      = os.environ.get("ZOHO_APP_NAME", "trrain")
ZOHO_DATA_CENTER   = os.environ.get("ZOHO_DATA_CENTER", "in")
ZOHO_REDIRECT_URI  = os.environ.get("ZOHO_REDIRECT_URI", "")

# Report / form link names
ZOHO_STUDENT_REPORT   = os.environ.get("ZOHO_STUDENT_REPORT",   "CV_Management")
ZOHO_ATTENDANCE_FORM  = os.environ.get("ZOHO_ATTENDANCE_FORM",  "Attendance_form")
ZOHO_ATTENDANCE_REPORT = os.environ.get("ZOHO_ATTENDANCE_REPORT", "Attendance_form_Report")

# ─── Student Database field names ─────────────────────────────────────────────
FIELD_STUDENT_ID        = "ID"   # Zoho system record ID — always present
FIELD_STUDENT_NUMBER    = os.environ.get("FIELD_STUDENT_NUMBER",    "Registration_No")
FIELD_STUDENT_NAME      = os.environ.get("FIELD_STUDENT_NAME",      "Name")
FIELD_STUDENT_PHOTO     = os.environ.get("FIELD_STUDENT_PHOTO",     "Upload_Photo1")

# Multi-line text field to cache the pre-computed 512-d ArcFace embedding (JSON list)
# Add this field in Zoho Creator: Student Database → Multi Line field → link name: Face_Embedding
FIELD_STUDENT_EMBEDDING = os.environ.get("FIELD_STUDENT_EMBEDDING", "Face_Embedding")

# ─── Attendance form field names (Attendance_form) ────────────────────────────
FIELD_ATT_TRAINEE_REG  = os.environ.get("FIELD_ATT_TRAINEE_REG",  "Trainee_Regstration") # lookup
FIELD_ATT_DATE         = os.environ.get("FIELD_ATT_DATE",         "Attendance_Date")
FIELD_ATT_STATUS       = os.environ.get("FIELD_ATT_STATUS",       "Attendance_Status")   # dropdown: Present
FIELD_ATT_FINANCIAL_YR = os.environ.get("FIELD_ATT_FINANCIAL_YR", "Financial_Year")       # lookup
FIELD_ATT_ZONE         = os.environ.get("FIELD_ATT_ZONE",         "Zone")                 # lookup
FIELD_ATT_CENTRE       = os.environ.get("FIELD_ATT_CENTRE",       "Centres")              # lookup
FIELD_ATT_BATCH        = os.environ.get("FIELD_ATT_BATCH",        "Batches")              # lookup
FIELD_ATT_CHECKED_OUT  = os.environ.get("FIELD_ATT_CHECKED_OUT",  "Checked_out")          # dropdown: Yes/No
FIELD_ATT_SOURCE       = os.environ.get("FIELD_ATT_SOURCE",       "Source")               # fixed text
FIELD_ATT_VALUE        = os.environ.get("FIELD_ATT_VALUE",        "Value")                # fixed integer
FIELD_ATT_CAPTURE      = os.environ.get("FIELD_ATT_CAPTURE",      "Live_Captured_Image")  # image upload field
FIELD_CHECK_IN         = os.environ.get("FIELD_CHECK_IN",         "Check_In")             # time field
FIELD_CHECK_OUT        = os.environ.get("FIELD_CHECK_OUT",        "Check_Out")            # time field

# ─── Zoho Creator Environment ────────────────────────────────────────────────
# Overridden per-request by envUrlFragment from the Widget SDK.
# Set this env var if you want a server-side default other than production.
ZOHO_ENVIRONMENT = os.environ.get("ZOHO_ENVIRONMENT", "")   # "" = production (default)

# ─── Face Recognition Settings ────────────────────────────────────────────────
try:
    FACE_MATCH_TOLERANCE = float(os.environ.get("FACE_MATCH_TOLERANCE", "0.40"))
    if not (0.20 <= FACE_MATCH_TOLERANCE <= 0.80):
        _cfg_logger.critical(
            f"FACE_MATCH_TOLERANCE={FACE_MATCH_TOLERANCE} is outside safe range [0.20, 0.80]. "
            "Clamping to 0.40 — set a value between 0.20 and 0.80."
        )
        FACE_MATCH_TOLERANCE = 0.40
except (ValueError, TypeError):
    _cfg_logger.critical("FACE_MATCH_TOLERANCE is not a valid float — using default 0.40.")
    FACE_MATCH_TOLERANCE = 0.40

try:
    CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))
except (ValueError, TypeError):
    _cfg_logger.critical("CACHE_TTL_SECONDS is not a valid integer — using default 86400.")
    CACHE_TTL_SECONDS = 86400

# ─── Batch filtering (ongoing batches only) ──────────────────────────────────
ZOHO_BATCHES_REPORT   = os.environ.get("ZOHO_BATCHES_REPORT",   "All_Batches")
FIELD_BATCH_STATUS    = os.environ.get("FIELD_BATCH_STATUS",    "Batch_Status")
FIELD_BATCH_CENTER    = os.environ.get("FIELD_BATCH_CENTER",    "Centres")
FIELD_STUDENT_BATCH   = os.environ.get("FIELD_STUDENT_BATCH",   "Batch_ID")
# Human-readable batch identifier field in the Batches form (e.g. "PKGJAHMJSS2672409")
FIELD_BATCH_DISPLAY   = os.environ.get("FIELD_BATCH_DISPLAY",   "Batch_ID")
# Start/end date fields in the Batches form — used to detect completed batches
FIELD_BATCH_START_DATE = os.environ.get("FIELD_BATCH_START_DATE", "Batch_Start_Date")
FIELD_BATCH_END_DATE   = os.environ.get("FIELD_BATCH_END_DATE",   "Batch_End_Date")

# ─── Centres report (for center-scoped student lookup) ───────────────────────
# Report link name of the Centres report in Zoho Creator
ZOHO_CENTRES_REPORT      = os.environ.get("ZOHO_CENTRES_REPORT",      "All_Centres")
# Field link name of the login email field in the Centres form
FIELD_CENTRE_LOGIN_EMAIL = os.environ.get("FIELD_CENTRE_LOGIN_EMAIL", "Email")
# Field link name of the centre display name in the Centres form
FIELD_CENTRE_NAME        = os.environ.get("FIELD_CENTRE_NAME",        "Centre_Name")
# Field link name of the Zone lookup in the Centres form (used for attendance Zone lookup)
FIELD_CENTRE_ZONE        = os.environ.get("FIELD_CENTRE_ZONE",        "Select_Zone")
# Zones report and name field (used to resolve zone name → zone record ID)
ZOHO_ZONES_REPORT        = os.environ.get("ZOHO_ZONES_REPORT",        "All_Zones")
FIELD_ZONE_NAME          = os.environ.get("FIELD_ZONE_NAME",           "Name")
# Field link name of the Center lookup in the student database
FIELD_STUDENT_CENTER = os.environ.get("FIELD_STUDENT_CENTER", "Centre_Name")
# Field link name of the Financial Year lookup in the student database
FIELD_STUDENT_FY     = os.environ.get("FIELD_STUDENT_FY",     "Financial_Year")

# ─── User Management (feature-flag gate) ─────────────────────────────────────
# Report that holds one record per Zoho user with feature-flag fields.
# Used by the Widget SDK to check if Face Recognition is enabled for the user.
ZOHO_USER_MGMT_REPORT     = os.environ.get("ZOHO_USER_MGMT_REPORT",     "All_Users")
FIELD_USER_MGMT_EMAIL     = os.environ.get("FIELD_USER_MGMT_EMAIL",     "Zoho_ID")
FIELD_USER_FACE_FEATURE   = os.environ.get("FIELD_USER_FACE_FEATURE",   "Face_Recognition_Feature")

# ─── App Settings ─────────────────────────────────────────────────────────────
try:
    PORT = int(os.environ.get("PORT", 5000))
except (ValueError, TypeError):
    PORT = 5000

DEBUG      = os.environ.get("DEBUG", "false").lower() == "true"
_IS_RENDER = bool(os.environ.get("RENDER"))   # True when running on Render

_SECRET_KEY_DEFAULT = "change-this-secret-key-in-production"
SECRET_KEY = os.environ.get("SECRET_KEY", _SECRET_KEY_DEFAULT)
if SECRET_KEY == _SECRET_KEY_DEFAULT:
    msg = (
        "SECRET_KEY is using the insecure default value. "
        "Set a strong random secret in your Render environment variables immediately."
    )
    if _IS_RENDER and not DEBUG:
        import sys as _sys
        _cfg_logger.critical(msg + " Refusing to start in production with insecure default.")
        _sys.exit(1)
    else:
        _cfg_logger.critical(msg)

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
_ADMIN_SECRET_DEFAULT = "train-admin-2026"
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", _ADMIN_SECRET_DEFAULT)
if not ADMIN_SECRET:
    import sys as _sys
    _cfg_logger.critical("ADMIN_SECRET is empty — refusing to start. Set it in Render environment variables.")
    _sys.exit(1)
elif ADMIN_SECRET == _ADMIN_SECRET_DEFAULT:
    _msg = (
        "ADMIN_SECRET is using the publicly known default 'train-admin-2026'. "
        "Set a strong random secret in your Render environment variables immediately."
    )
    if _IS_RENDER and not DEBUG:
        import sys as _sys
        _cfg_logger.critical(_msg + " Refusing to start in production with insecure default.")
        _sys.exit(1)
    else:
        _cfg_logger.critical(_msg)
