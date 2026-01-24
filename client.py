import os
import time
import requests

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8000/Get_Inference"
REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "GNR_hi.wav")
OUTPUT_DIR = "client_outputs"
TIMEOUT = 120
TIMEOUT_LONG = 180

os.makedirs(OUTPUT_DIR, exist_ok=True)

# One short phrase per language (from datasets where available; fallbacks for ta, bh)
LANGUAGE_EXAMPLES = {
    "hi": "नमस्ते, आप कैसे हैं, today आपका दिन कैसा जा रहा है और सब कुछ ठीक तो है?",
    "kn": "ಲಕ್ಷ್ಮೀ ಇಂದು ಸಾಕಷ್ಟು ಹಾಲು ಕೊಟ್ಟಳು, ಅದನ್ನು ನೋಡಿ family ಗೆ ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದು God ಗೆ thank you ಹೇಳಿದರು.",
    "bh": "ई एक ठोस परीक्षण बा, जवन clearly देखावता कि experiment सही तरीका से कइल गइल बा।",
    "mr": "शेजारी त्याची मनापासून स्तुती करीत होता, कारण त्याने वेळेवर सगळ्यांच्या मदतीला support केला.",
    "mai": "की अहाँसभ नव छात्रसभ केँ सेहो scholarship दैत छी, जाहि सँ गरीब परिवारक विद्यार्थी पढ़ाई continue रखि सकथि?",
    "mag": "मौसम विज्ञान के पढाई गणेशबा एही से कैले हय कि ऊ farming आउ environment के बारे में deep समझ चाह रहल हय।",
    "gu": "એણે બધાની સામે સ્મિત સાથે ‘નમસ્તે’ કહ્યું અને પછી friendly વાતચીત start કરી.",
    "bn": "গত চব্বিশ ঘণ্টায় একটানা বৃষ্টি হয়েছে, total পরিমাণ ছিল একত্রিশ মিলিমিটার, যার ফলে weather একটু ঠান্ডা হয়েছে।",
    "hne": "अत्या पात्य ह एकठन पारंपरिक भारतीय tag खेल हरय, जेन ला बचपन म गाँव के मैदान म सब friends मिलके खेलथें।"
}

def _is_valid_wav(data: bytes) -> bool:
    if len(data) <= 44:
        return False
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def run_test(session, test_name, text, lang=None, expect_success=True, method="POST", timeout=TIMEOUT):
    """Send request, check status, and for 200: validate audio/wav and WAV header. lang is optional."""
    preview = text[:50] + "..." if len(text) > 50 else text
    print(f"\n🔹 Running Test: {test_name}")
    print(f"   Input Text: '{preview}'")
    print(f"   Method: {method}")

    params = {"text": text}
    if lang is not None:
        params["lang"] = lang

    start = time.time()
    try:
        with open(REF_AUDIO_PATH, "rb") as f:
            files = {"speaker_wav": f}
            if method == "GET":
                resp = session.get(BASE_URL, params=params, files=files, timeout=timeout)
            else:
                resp = session.post(BASE_URL, params=params, files=files, timeout=timeout)
        elapsed = time.time() - start
        print(f"   ⏱ {elapsed:.2f}s")

        if expect_success:
            if resp.status_code == 200:
                ct = (resp.headers.get("Content-Type") or "").lower()
                ok_type = "audio" in ct or "audio/wav" in ct
                ok_wav = _is_valid_wav(resp.content)
                if ok_type and ok_wav:
                    out = os.path.join(OUTPUT_DIR, f"{test_name}.wav")
                    with open(out, "wb") as fp:
                        fp.write(resp.content)
                    print(f"   PASS: Audio ({len(resp.content)} bytes) -> {out}")
                else:
                    print(f"   FAIL: 200 but invalid: Content-Type={ct!r}, valid WAV={ok_wav}")
            else:
                print(f"   FAIL: Expected 200, got {resp.status_code}")
                print(f"   Reason: {resp.text}")
        else:
            if resp.status_code != 200:
                print(f"   PASS: Correctly failed with {resp.status_code} as expected.")
                print(f"   Error Msg: {resp.text}")
            else:
                print(f"   FAIL: Expected failure, but got 200 OK.")

    except requests.exceptions.ConnectionError:
        elapsed = time.time() - start
        print(f"   {elapsed:.2f}s")
        print("   CRITICAL FAIL: Could not connect to server. Is 'server.py' running?")

# ==========================================
# TEST CASES
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(REF_AUDIO_PATH):
        raise FileNotFoundError(f"Ref audio not found: {REF_AUDIO_PATH}")

    total_start = time.time()
    session = requests.Session()

    try:
        # --- Per-language tests (LANGUAGE_EXAMPLES) ---
        for lang, text in LANGUAGE_EXAMPLES.items(): 
            run_test(session, f"lang_{lang}", text, lang=lang)

        # --- Long-text stress ---
        long_text = "'बंधन' शांतनु, सत्यवती तथा भीष्म के मनोविज्ञान तथा जीवन-मूल्यों की कथा है। घटनाओं की दृष्टि से यह सत्यवती के हस्तिनापुर में आने तथा हस्तिनापुर से चले जाने के मध्य की अवधि की कथा है, जिसमें जीवन के उच्च आध्यात्मिक मूल्य जीवन की निम्नता और भौतिकता के सम्मुख असमर्थ होते महासमर-बंधन प्रतीत होते हैं और हस्तिनापुर का जीवन महाभारत के युद्ध की दिशा ग्रहण करने लगता है। महासमर-बंधन (खंड एक) स्पष्ट तौर पर दिखाता है कि  किस प्रकार शांतनु, सत्यवती तथा भीष्म के महासमर-बंधन कर्म-बन्धनों से हस्तिनापुर बँध चुका है और भीष्म भी उससे मुक्त होने की स्थिति में नहीं थे।"
        run_test(session, "long_text", long_text, timeout=TIMEOUT_LONG)

        # --- POST method (lang omitted) ---
        run_test(session, "post_method", LANGUAGE_EXAMPLES["hi"], method="POST")

        # --- Missing text (negative, expect 422) ---
        print("\n🔹 Running Test: missing_text (Expect 422 Error)")
        t0 = time.time()
        try:
            with open(REF_AUDIO_PATH, "rb") as f:
                r = session.post(BASE_URL, params={}, files={"speaker_wav": f}, timeout=TIMEOUT)
            print(f"   ⏱ {time.time() - t0:.2f}s")
            if r.status_code == 422:
                print("   PASS: Server correctly rejected missing text.")
            else:
                print(f"   FAIL: Expected 422, got {r.status_code}")
        except Exception as e:
            print(f"   ⏱ {time.time() - t0:.2f}s")
            print(f"   Error: {e}")

    finally:
        print(f"\n All tests completed in {time.time() - total_start:.2f}s. Check '{OUTPUT_DIR}' for audio files.")