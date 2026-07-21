# Custom API — Attendance Check-In / Check-Out with Photo (2-call design)

> AI-assisted (Cowork). **Sandbox-test both functions and peer-review before merge/deploy.**

## Goal

Collapse today's **4 REST calls** into **2 custom API calls**, invoked from the widget by portal users.

| Event | Today | New (1 call each) |
|---|---|---|
| Check-in | create record + upload check-in photo | `Post_with_Img` → insert record **and** attach `Check_In_Image` |
| Check-out | PATCH checkout + (new) upload checkout photo | `Check_Out_with_Img` → update record **and** attach `Check_Out_Image` |

The image can't ride the insert/update directly (Zoho Add/Update Record APIs reject image fields), so each function decodes the base64 photo and pushes it to the image field via the `/upload` endpoint using a connection — the same mechanism as the existing `_upload_capture_photo`, but server-side.

---

## Prerequisites (do these first)

1. **Fields on `Attendance_form`** — add two Image fields (replacing the single `Live_Captured_Image`):
   - `Check_In_Image` (Image)
   - `Check_Out_Image` (Image) — *new*
2. **Connection** — Microservices → Connections → **+ Create New** → **Zoho OAuth**.
   - Name/link name: **`creator`** (must match the `connection:` in the code)
   - Scopes: `ZohoCreator.report.CREATE` (upload) and `ZohoCreator.report.UPDATE`
   - Create & Authorize, and make sure it's enabled for the `trrain` app.
3. **Standalone Deluge functions** — create the two below (Workflow → Functions), *then* select them on the custom API **Actions** tab.

---

## Field mapping (check-in insert)

| Form field (link name) | Type | Value |
|---|---|---|
| `Attendance_Date` | Date | today (or passed) |
| `Trainee_Regstration` | Lookup | trainee record ID |
| `Attendance_Status` | Dropdown | `Present` |
| `Value` | Number | `1` (Present) / `0` (Absent) |
| `Zone` | Lookup | zone ID |
| `Centres` | Lookup | centre ID |
| `Batches` | Lookup | batch ID |
| `Check_In` | Time | `HH:mm:ss` |
| `Checked_out` | Dropdown | `No` |
| `Source` | Dropdown | `Live Face Recognition` |
| `Check_In_Image` | Image | live photo (via upload) |

---

## Function 1 — Check-In (attach to the "Post with Img" custom API)

```deluge
map AttendanceAPI.checkInWithPhoto(string trainee_id, string zone_id, string centre_id, string batch_id, string check_in_time, string image_base64)
{
	result = Map();

	// ---- Guard: required ----
	if(trainee_id == null || trainee_id.trim() == "")
	{
		result.put("code",4001);
		result.put("message","trainee_id is required");
		return result;
	}

	// ---- Normalise time to HH:mm:ss ----
	ciTime = check_in_time;
	if(ciTime != null && ciTime.length() == 5)
	{
		ciTime = ciTime + ":00";
	}

	// ---- Insert attendance record (image attached separately below) ----
	// NOTE: lookups (Zone/Centres/Batches) assume non-empty IDs. If any can be
	// blank, move that line into a conditional `update` after the insert.
	recID = insert into Attendance_form
	[
		Attendance_Date = zoho.currentdate
		Trainee_Regstration = trainee_id
		Attendance_Status = "Present"
		Value = 1
		Zone = zone_id
		Centres = centre_id
		Batches = batch_id
		Check_In = ciTime
		Checked_out = "No"
		Source = "Live Face Recognition"
	];
	result.put("record_id",recID);

	// ---- Attach check-in photo ----
	if(image_base64 != null && image_base64.trim() != "")
	{
		b64 = image_base64;
		if(b64.contains(","))                       // strip data-URI prefix if present
		{
			b64 = b64.subString(b64.indexOf(",") + 1);
		}
		imgFile = zoho.encryption.base64DecodeToFile(b64,"checkin_" + recID + ".jpg");
		imgFile.setParamName("file");
		uploadResp = invokeurl
		[
			url : "https://www.zohoapis.in/creator/v2.1/data/admin_trrainfoundation/trrain/report/Attendance_form_Report/" + recID + "/Check_In_Image/upload"
			type : POST
			files : imgFile
			connection : "creator"
		];
		result.put("photo_upload",uploadResp);
	}

	result.put("code",3000);
	result.put("message","Check-in recorded");
	return result;
}
```

**Actions tab args (in this order):** `trainee_id, zone_id, centre_id, batch_id, check_in_time, image_base64`

---

## Function 2 — Check-Out (build a second custom API for this)

```deluge
map AttendanceAPI.checkOutWithPhoto(string record_id, string check_out_time, string image_base64)
{
	result = Map();
	if(record_id == null || record_id.trim() == "")
	{
		result.put("code",4001);
		result.put("message","record_id is required");
		return result;
	}

	coTime = check_out_time;
	if(coTime != null && coTime.length() == 5)
	{
		coTime = coTime + ":00";
	}

	update Attendance_form [ID == record_id.toLong()]
	[
		Check_Out = coTime
		Checked_out = "Yes"
	];

	if(image_base64 != null && image_base64.trim() != "")
	{
		b64 = image_base64;
		if(b64.contains(","))
		{
			b64 = b64.subString(b64.indexOf(",") + 1);
		}
		imgFile = zoho.encryption.base64DecodeToFile(b64,"checkout_" + record_id + ".jpg");
		imgFile.setParamName("file");
		uploadResp = invokeurl
		[
			url : "https://www.zohoapis.in/creator/v2.1/data/admin_trrainfoundation/trrain/report/Attendance_form_Report/" + record_id + "/Check_Out_Image/upload"
			type : POST
			files : imgFile
			connection : "creator"
		];
		result.put("photo_upload",uploadResp);
	}

	result.put("code",3000);
	result.put("record_id",record_id);
	result.put("message","Check-out recorded");
	return result;
}
```

**Actions tab args:** `record_id, check_out_time, image_base64`

---

## Widget invocation (base64 in JSON payload)

Get a base64 JPEG from your capture canvas, then call the custom API. `content_type` stays `application/json` (only mode that carries file data from a widget).

```javascript
// canvas -> base64 (strip the "data:image/jpeg;base64," prefix)
const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
const imageBase64 = dataUrl.split(",")[1];

// ----- CHECK-IN -----
ZOHO.CREATOR.init().then(function () {
  ZOHO.CREATOR.DATA.invokeCustomApi({
    api_name: "Post_with_Img",          // your custom API link name
    workspace_name: "trrain",
    http_method: "POST",
    content_type: "application/json",
    payload: {
      trainee_id:    traineeRecordId,
      zone_id:       zoneId,
      centre_id:     centreId,
      batch_id:      batchId,
      check_in_time: "09:15:00",
      image_base64:  imageBase64
    }
  }).then(function (r) {
    console.log("check-in", r);          // r.result.record_id -> keep for checkout
  });
});

// ----- CHECK-OUT (later) -----
ZOHO.CREATOR.DATA.invokeCustomApi({
  api_name: "Check_Out_with_Img",
  workspace_name: "trrain",
  http_method: "POST",
  content_type: "application/json",
  payload: {
    record_id:      savedRecordId,       // from the check-in response
    check_out_time: "17:30:00",
    image_base64:   checkoutImageBase64
  }
});
```

---

## Sandbox test checklist

- [ ] Connection `creator` authorised & enabled for `trrain`.
- [ ] `Check_In_Image` / `Check_Out_Image` fields exist on `Attendance_form`.
- [ ] Check-in: record created with all fields + photo visible in `Check_In_Image`.
- [ ] Check-out: same record updated, `Checked_out=Yes`, `Check_Out` set, `Check_Out_Image` visible.
- [ ] Test with a blank lookup (e.g. no batch) — confirm insert doesn't fail; if it does, switch that lookup to a conditional post-insert `update`.
- [ ] Confirm portal user can invoke (User Scope = Portal users) and the user is an approved portal user.
- [ ] Check API logs (Microservices → Custom API → hits) for success code 3000.
```
