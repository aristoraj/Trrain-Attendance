# Graph Report - .  (2026-06-03)

## Corpus Check
- Corpus is ~37,893 words - fits in a single context window. You may not need a graph.

## Summary
- 479 nodes · 876 edges · 32 communities (24 shown, 8 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 33 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Attendance Queue Core|Attendance Queue Core]]
- [[_COMMUNITY_Face Model and Config|Face Model and Config]]
- [[_COMMUNITY_Frontend and Docs|Frontend and Docs]]
- [[_COMMUNITY_Flask API and Webhooks|Flask API and Webhooks]]
- [[_COMMUNITY_Zoho Widget Frontend|Zoho Widget Frontend]]
- [[_COMMUNITY_Admin API Endpoints|Admin API Endpoints]]
- [[_COMMUNITY_Queue Drain and Sync|Queue Drain and Sync]]
- [[_COMMUNITY_Desk Widget Plugin|Desk Widget Plugin]]
- [[_COMMUNITY_Zoho API and Embeddings|Zoho API and Embeddings]]
- [[_COMMUNITY_Config Unit Tests|Config Unit Tests]]
- [[_COMMUNITY_Student Cache Layer|Student Cache Layer]]
- [[_COMMUNITY_Batch and Background Load|Batch and Background Load]]
- [[_COMMUNITY_Anti-Spoof Liveness|Anti-Spoof Liveness]]
- [[_COMMUNITY_Cache Init and Startup|Cache Init and Startup]]
- [[_COMMUNITY_Test Fixtures|Test Fixtures]]
- [[_COMMUNITY_Session Token Auth|Session Token Auth]]
- [[_COMMUNITY_Attendance Posting|Attendance Posting]]
- [[_COMMUNITY_Brand Assets|Brand Assets]]
- [[_COMMUNITY_Claude Code Config|Claude Code Config]]
- [[_COMMUNITY_Deployment and CI|Deployment and CI]]
- [[_COMMUNITY_Git Automation|Git Automation]]
- [[_COMMUNITY_Dedup Check|Dedup Check]]
- [[_COMMUNITY_SDK Posted Dedup|SDK Posted Dedup]]
- [[_COMMUNITY_Widget Sample App|Widget Sample App]]
- [[_COMMUNITY_Widget Translations|Widget Translations]]
- [[_COMMUNITY_Widget Resources JSON|Widget Resources JSON]]
- [[_COMMUNITY_GitHub Push Script|GitHub Push Script]]

## God Nodes (most connected - your core abstractions)
1. `AttendanceQueue` - 65 edges
2. `str` - 32 edges
3. `ZohoCreatorAPI` - 29 edges
4. `FaceCache` - 18 edges
5. `str` - 16 edges
6. `verify()` - 15 edges
7. `str` - 15 edges
8. `_resolve_env()` - 13 edges
9. `int` - 13 edges
10. `embedding_to_json()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `AttendanceQueue.get_daily_cache (KV 24h TTL cache)` --semantically_similar_to--> `get_user_centers_cached()`  [INFERRED] [semantically similar]
  attendance_queue.py → app.py
- `FaceCache` --semantically_similar_to--> `Multi-Tier Cache (L1 in-memory + L2 PostgreSQL + L3 Zoho API)`  [INFERRED] [semantically similar]
  face_utils.py → app.py
- `Webcam Face Attendance UI` --shares_data_with--> `ZohoCreatorAPI`  [INFERRED]
  static/index.html → zoho_api.py
- `Project README and Architecture Overview` --references--> `ZohoCreatorAPI`  [EXTRACTED]
  README.md → zoho_api.py
- `test_verify_with_expired_token_returns_401()` --calls--> `_issue_session_token()`  [EXTRACTED]
  tests/test_api.py → app.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Face Verification Pipeline (decode → detect → liveness → match → queue)** — app_verify, face_utils_decode_base64_image, face_utils_encode_face_with_bbox, liveness_utils_check_liveness, face_utils_find_best_match, attendance_queue_enqueue_if_not_marked [EXTRACTED 0.95]
- **Cold Start DB Restore (startup → DB → scope caches populated without Zoho API call)** — app_restore_face_caches_from_db, attendance_queue_load_students_from_db, face_utils_json_to_embedding, face_utils_facecache [INFERRED 0.90]
- **Webhook Encode-and-Inject Flow (webhook → encode → cache update + DB name sync)** — app_webhook_student_update, app_inject_or_update_student_in_caches, attendance_queue_update_student_name_everywhere, attendance_queue_upsert_student_in_scope [EXTRACTED 0.95]
- **Face Embedding Lifecycle: Download → Encode → Store → Cache** — zoho_api_download_and_encode, zoho_api_save_embedding, zoho_api_process_record, zoho_api_encode_and_save_to_creator [INFERRED 0.85]
- **Zoho OAuth Token Refresh Chain** — zoho_api_get_token, zoho_api_refresh_token, zoho_api_headers, zoho_api_request [EXTRACTED 1.00]
- **face_utils Test Suite (TC-006 to TC-016)** — tests_test_face_utils_test_encode_face_from_array_no_face, tests_test_face_utils_test_find_best_match_exact_hit, tests_test_face_utils_test_embedding_round_trip, tests_test_face_utils_test_face_cache_ttl_expiry, tests_test_face_utils_test_get_face_app_singleton_concurrency [EXTRACTED 1.00]

## Communities (32 total, 8 thin omitted)

### Community 0 - "Attendance Queue Core"
Cohesion: 0.06
Nodes (45): AttendanceQueue, _ConnWrapper, AttendanceQueue._init_db (schema creation + migrations), bool, float, int, str, SQLite / PostgreSQL attendance outbox with async Zoho sync.  Flow per verify req (+37 more)

### Community 1 - "Face Model and Config"
Cohesion: 0.06
Nodes (46): _warmup_face_model(), bytes, Configuration for Zoho Face Recognition Module. All values are loaded from envi, decode_base64_image(), embedding_to_json(), encode_face_from_array(), encode_face_from_bytes(), encode_face_with_bbox() (+38 more)

### Community 2 - "Frontend and Docs"
Cohesion: 0.11
Nodes (20): Project README and Architecture Overview, Response, Webcam Face Attendance UI, bool, str, Authenticated HTTP request with one automatic retry on 401 (token refresh)., Look up the logged-in user's centres by querying the All_Centres report, Return Zoho record IDs of all Ongoing batches that belong to the given centers. (+12 more)

### Community 3 - "Flask API and Webhooks"
Cohesion: 0.07
Nodes (29): admin_sync_status(), Flask App (main entry, CORS, rate limiter), health(), Shows attendance queue health — pending/posted/failed counts and failed records., Called by a Zoho Creator Deluge workflow whenever a Trainee record is     create, webhook_student_update(), AttendanceQueue.get_status_summary (pending/posted/failed counts), AttendanceQueue.update_student_name_everywhere (cross-scope name sync) (+21 more)

### Community 4 - "Zoho Widget Frontend"
Cohesion: 0.07
Nodes (29): dependencies, body-parser, express, react, react-dom, zd-styles, devDependencies, @babel/core (+21 more)

### Community 5 - "Admin API Endpoints"
Cohesion: 0.09
Nodes (26): admin_clear_daily_cache(), admin_clear_student_embeddings(), admin_clear_today(), admin_encode_all_students(), admin_login(), admin_reauth_page(), admin_reauth_submit(), admin_retry_failed() (+18 more)

### Community 6 - "Queue Drain and Sync"
Cohesion: 0.07
Nodes (27): AttendanceQueue.add_verified_embedding (rotating slot save), AttendanceQueue._drain (batch process PENDING rows), AttendanceQueue._drain_loop (background sync worker), AttendanceQueue.enqueue_if_not_marked (atomic dedup+enqueue), AttendanceQueue._handle_failure (exponential backoff), AttendanceQueue.save_local_embedding (upsert embedding), Attendance Outbox Pattern (enqueue→sync→dedup), Unit tests for attendance_queue.py Covers TC-031 to TC-039 from the QA test plan (+19 more)

### Community 7 - "Desk Widget Plugin"
Cohesion: 0.09
Nodes (22): Zoho Desk Widget Extension JS (bundled output), Zoho Widget NPM Package (zoho-app), config, connectors, cspDomains, connect-src, locale, modules (+14 more)

### Community 8 - "Zoho API and Embeddings"
Cohesion: 0.13
Nodes (19): ZohoCreatorAPI.check_duplicate_attendance, ZohoCreatorAPI._download_and_encode, Embedding Priority Strategy Rationale, ZohoCreatorAPI.encode_and_save_to_creator, ZohoCreatorAPI._extract_photo_url, ZohoCreatorAPI.get_ongoing_batch_ids, ZohoCreatorAPI.get_students, ZohoCreatorAPI.get_students_list (+11 more)

### Community 9 - "Config Unit Tests"
Cohesion: 0.16
Nodes (17): Unit tests for config.py validation logic. Covers TC-002, TC-003, TC-004 from th, Helper: reload config.py with specific env vars., FACE_MATCH_TOLERANCE=2.0 → clamped to 0.40., FACE_MATCH_TOLERANCE=-0.5 → clamped to 0.40., Valid FACE_MATCH_TOLERANCE=0.55 → accepted as-is., CACHE_TTL_SECONDS='abc' → falls back to 86400 without crash., CACHE_TTL_SECONDS='86400.5' → falls back to 86400 (int() fails on floats)., Valid CACHE_TTL_SECONDS='3600' → accepted. (+9 more)

### Community 10 - "Student Cache Layer"
Cohesion: 0.24
Nodes (17): _build_scope_key(), cache_refresh(), _get_cache(), get_context(), get_students_cached(), get_user_centers_cached(), preload_students(), str (+9 more)

### Community 11 - "Batch and Background Load"
Cohesion: 0.18
Nodes (14): get_batch_ids_cached(), _load_students_bg(), Background worker: load + cache students without blocking an HTTP request., Returns (batch_ids, batch_names) both cached for 24h., AttendanceQueue.get_daily_cache (KV 24h TTL cache), AttendanceQueue.is_scope_fully_catalogued (check flag), AttendanceQueue.mark_scope_catalogued (no-expiry flag), AttendanceQueue.remove_students_by_batch (completed batch cleanup) (+6 more)

### Community 12 - "Anti-Spoof Liveness"
Cohesion: 0.22
Nodes (12): check_liveness(), _crop_face(), _get_session(), bool, float, ndarray, str, Passive face liveness detection using MiniFASNetV2 (ONNX).  Detects screen/video (+4 more)

### Community 13 - "Cache Init and Startup"
Cohesion: 0.20
Nodes (11): index(), _inject_or_update_student_in_caches(), load_students(), Accept raw Zoho Creator records fetched by the Widget SDK and seed the face cach, On startup, rebuild FaceCaches from local DB so the app serves verify     reques, Insert or update a student in all warm in-memory scope caches that match centre_, _restore_face_caches_from_db(), AttendanceQueue.get_local_embeddings (multi-source fetch) (+3 more)

### Community 14 - "Test Fixtures"
Cohesion: 0.20
Nodes (9): blank_rgb_image(), dummy_embedding(), dummy_student(), Shared fixtures for the test suite., A normalised 512-d random embedding vector., A minimal student dict as returned by _process_record., 200×200 blank white RGB numpy array (no face)., AttendanceQueue backed by a temp SQLite DB (no PostgreSQL needed). (+1 more)

### Community 15 - "Session Token Auth"
Cohesion: 0.22
Nodes (9): create_session(), feature_access(), _get_feature_access(), _issue_session_token(), bool, Issue a short-lived session token AND check feature-access in one call.      Ver, Core logic: returns True if email has Face_Recognition_Feature enabled., Check Face_Recognition_Feature flag. Requires session auth (use /api/session for (+1 more)

### Community 16 - "Attendance Posting"
Cohesion: 0.33
Nodes (6): post_attendance(), Verify signature and expiry. Returns payload dict or None., Decorator: require a valid widget session token (Bearer in Authorization header), Server-side attendance posting fallback.     Called by the frontend when SDK add, require_session(), _verify_session_token()

### Community 17 - "Brand Assets"
Cohesion: 0.90
Nodes (5): App Icon - Green Communication Bubble with Checkmark, Checkmark Symbol - Confirmation/Completion Concept, Green Chat Bubble Symbol - Communication/Messaging Concept, Phone Handset Symbol - Telephony/Call Concept, Brand Logo - Green Communication Bubble with Checkmark

### Community 19 - "Deployment and CI"
Cohesion: 1.00
Nodes (3): Render Deployment Configuration, Python Dependencies, GitHub Actions CI Pipeline

## Knowledge Gaps
- **78 isolated node(s):** `PreToolUse`, `name`, `version`, `private`, `start` (+73 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AttendanceQueue` connect `Attendance Queue Core` to `Admin API Endpoints`, `Student Cache Layer`, `Cache Init and Startup`, `Test Fixtures`, `Session Token Auth`?**
  _High betweenness centrality (0.276) - this node is a cross-community bridge._
- **Why does `ZohoCreatorAPI` connect `Frontend and Docs` to `Face Model and Config`, `Admin API Endpoints`, `Zoho API and Embeddings`, `Student Cache Layer`, `Cache Init and Startup`, `Session Token Auth`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Why does `verify()` connect `Student Cache Layer` to `Face Model and Config`, `Flask API and Webhooks`, `Admin API Endpoints`, `Queue Drain and Sync`, `Anti-Spoof Liveness`, `Attendance Posting`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `AttendanceQueue` (e.g. with `bool` and `str`) actually correct?**
  _`AttendanceQueue` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `ZohoCreatorAPI` (e.g. with `bool` and `str`) actually correct?**
  _`ZohoCreatorAPI` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FaceCache` (e.g. with `cache_refresh()` and `bool`) actually correct?**
  _`FaceCache` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PreToolUse`, `name`, `version` to the rest of the system?**
  _215 weakly-connected nodes found - possible documentation gaps or missing edges._