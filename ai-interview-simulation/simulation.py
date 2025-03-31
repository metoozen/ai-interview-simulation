import os
import re
import json
import pyttsx3
import speech_recognition as sr
import openai
from groq import Client
from deep_translator import GoogleTranslator

# ------------------ AYARLAR ------------------
GROQ_API_KEY = "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"
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

client = Client(api_key=GROQ_API_KEY)

# JSON dosyalarından job post ve CV bilgilerini yükle
def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"Dosya '{filename}' bulunamadı.")
        return {}

job_post = load_json("job_post.json")
cv = load_json("cv.json")
employer = {
    "company": "MertTech Solutions",
    "profile": "Öncü teknolojik çözümler üreten, yapay zeka ve veri bilimi odaklı bir firma."
}

# Transcript dosyası adı
TRANSCRIPT_FILENAME = "transcript.json"

if os.path.exists(TRANSCRIPT_FILENAME):
    with open(TRANSCRIPT_FILENAME, "r", encoding="utf-8") as f:
        transcript = json.load(f)
else:
    transcript = {"messages": []}

def save_transcript():
    with open(TRANSCRIPT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

def add_to_transcript(role, content):
    transcript["messages"].append({"role": role, "content": content})
    save_transcript()

# Guideline: LLM'in mülakatı nasıl yöneteceğine dair kapsamlı direktifler.
GUIDELINE = (
    "You are an interviewer for an Artificial Intelligence Engineer role. "
    "Based on the provided job post and the candidate's CV, conduct a natural and technical interview. "
    "Your questions should cover the candidate's technical expertise, analytical thinking, problem-solving, and communication skills. "
    "Encourage detailed answers, ask follow-up questions as needed, and ensure each question explores a different aspect of the candidate's qualifications. "
    "Do not repeat previous questions. Refer only to the last two messages of the transcript for context. "
    "Avoid meta labels, off-topic content, or generic statements. "
    "Also, consider the candidate's experience as indicated in the CV when formulating questions."
)

# Yetkinlikler: İlk 2 soru minimum yeterlilikleri, sonraki 2 soru işin teknik detaylarını kapsayacak.
min_qualification_competencies = [
    ("Minimum Qualifications", "Assess if the candidate meets the minimum requirements based on the job post and CV. Consider their coursework, projects, and basic understanding of AI/ML concepts."),
    ("Job Fit", "Evaluate the candidate's understanding of the job responsibilities and how their background in the CV aligns with the role requirements.")
]
technical_competencies = [
    ("Technical Challenge", "Evaluate the candidate's ability to design and implement complex AI solutions. Focus on real-world problem solving and innovation."),
    ("Scalability and Optimization", "Assess the candidate's approach to scaling models, handling large datasets, and optimizing performance in practical scenarios.")
]
competencies = min_qualification_competencies + technical_competencies

# ------------------ YARDIMCI FONKSİYONLAR ------------------

def clean_meta_info(text):
    patterns = [
        r"^Bir adayın yapay zeka mühendisi rolündeki mesleki yeterliliğini değerlendirmek için teknik bir soru:\s*",
        r"Translated Question[:\-]*",
        r"Candidate's Response[:\-]*",
        r"\[.*?\]"
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def generate_ai_text(prompt, role="user"):
    refined_prompt = GUIDELINE + " " + prompt
    # Transcript geçmişinde yalnızca son 2 mesajı ekleyelim
    if len(transcript["messages"]) >= 2:
        history = " ".join([msg["content"] for msg in transcript["messages"][-2:]])
        refined_prompt += " Transcript History: " + history
    messages = [{"role": role, "content": refined_prompt}]
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    raw_response = response.choices[0].message.content
    return clean_meta_info(raw_response)

def translate_text(text, source="en", target="tr"):
    return GoogleTranslator(source=source, target=target).translate(text)

def record_candidate_response():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.pause_threshold = 2.0
        print("Lütfen yanıtınızı sesli olarak verin (Türkçe):")
        try:
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=90)
            text = recognizer.recognize_google(audio, language="tr-TR")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition sesi anlayamadı.")
            return ""
        except sr.RequestError as e:
            print(f"Google Speech Recognition hizmetine ulaşılamıyor; {e}")
            return ""

def evaluate_candidate(transcript):
    eval_prompt = (
        "Based on the following interview transcript, provide a comprehensive evaluation of the candidate's performance as an Artificial Intelligence Engineer. "
        "Assess their technical expertise, analytical thinking, problem-solving skills, and communication skills, as well as the depth and detail of their answers. "
        "Offer constructive feedback and suggestions for improvement. "
        "Transcript: " + " ".join([msg["content"] for msg in transcript["messages"]])
    )
    evaluation = generate_ai_text(eval_prompt, role="system")
    return evaluation

def generate_unique_question(prompt, role="user", max_attempts=3):
    last_question = ""
    for msg in reversed(transcript["messages"]):
        if msg["role"] == "user":
            last_question = msg["content"]
            break
    attempt = 0
    new_question = generate_ai_text(prompt, role=role)
    while last_question and is_similar(new_question, last_question) and attempt < max_attempts:
        attempt += 1
        new_question = generate_ai_text(prompt, role=role)
    return new_question

def is_similar(text1, text2, threshold=0.8):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if union else 0
    return similarity >= threshold

def interview_simulation():
    print("Mülakat simülasyonu başlatılıyor...\n")
    
    greeting = f"Merhaba, hoş geldiniz. Şirketimiz {employer['company']} olarak öncü teknolojik çözümler üretiyor. Şimdi mülakatımıza başlayalım."
    print("[AI Greeting]:", greeting)
    add_to_transcript("system", greeting)
    speak_text(greeting)

    general_intro = "Could you please briefly introduce yourself and describe your background?"
    print(f"\n[General Introduction Question]: {general_intro}")
    speak_text(translate_text(general_intro, source="en", target="tr"))
    candidate_intro = record_candidate_response()
    print(f"\n[Candidate Introduction]: {candidate_intro}\n")
    add_to_transcript("user", candidate_intro)
  
    for comp, description in competencies:
        prompt = f"{comp}: {description}"
        ai_question = generate_unique_question(prompt, role="user")
        print(f"\n[AI Question for {comp} in English]:\n{ai_question}\n")
        add_to_transcript("user", ai_question)
        
        turkish_question = translate_text(ai_question, source="en", target="tr")
        turkish_question = clean_meta_info(turkish_question)
        print(f"[Question for {comp} (Turkish)]:\n{turkish_question}\n")
        speak_text(turkish_question)
        
        candidate_response = record_candidate_response()
        print(f"\n[Candidate's Response for {comp} (Turkish)]:\n{candidate_response}\n")
        candidate_response_en = translate_text(candidate_response, source="tr", target="en")
        print(f"[Candidate's Response for {comp} Translated to English]:\n{candidate_response_en}\n")
        add_to_transcript("user", candidate_response_en)
        
        if len(candidate_response_en.split()) < 5:
            followup_prompt = f"The candidate's answer for {comp} seems brief. Can you ask a follow-up question to encourage more detailed response?"
            followup_question = generate_unique_question(followup_prompt, role="system")
            print(f"[Follow-up Question for {comp} in English]:\n{followup_question}\n")
            add_to_transcript("assistant", followup_question)
            turkish_followup = translate_text(followup_question, source="en", target="tr")
            turkish_followup = clean_meta_info(turkish_followup)
            print(f"[Follow-up Question for {comp} (Turkish)]:\n{turkish_followup}\n")
            speak_text(turkish_followup)
            candidate_followup = record_candidate_response()
            print(f"\n[Candidate's Follow-up Response for {comp} (Turkish)]:\n{candidate_followup}\n")
            candidate_followup_en = translate_text(candidate_followup, source="tr", target="en")
            print(f"[Candidate's Follow-up Response for {comp} Translated to English]:\n{candidate_followup_en}\n")
            add_to_transcript("user", candidate_followup_en)
            feedback_prompt = f"Based on the candidate's follow-up answer: '{candidate_followup_en}', provide constructive technical feedback regarding their {comp.lower()}."
        else:
            feedback_prompt = f"Based on the candidate's answer: '{candidate_response_en}', provide constructive technical feedback regarding their {comp.lower()}."
        ai_feedback = generate_ai_text(feedback_prompt, role="system")
        print(f"[AI Feedback for {comp}]:\n{ai_feedback}\n")
        add_to_transcript("assistant", ai_feedback)
    
    behavioral_question = "Could you tell me about a time when you faced a challenge working in a team and how you resolved it?"
    print(f"\n[Behavioral Question]: {behavioral_question}")
    speak_text(translate_text(behavioral_question, source="en", target="tr"))
    behavioral_response = record_candidate_response()
    print(f"\n[Candidate's Behavioral Response]: {behavioral_response}\n")
    add_to_transcript("user", behavioral_response)
    
    candidate_question_prompt = "Do you have any questions for us regarding the company or the role?"
    print(f"\n[Candidate Question Prompt]: {candidate_question_prompt}")
    speak_text(translate_text(candidate_question_prompt, source="en", target="tr"))
    candidate_question = record_candidate_response()
    print(f"\n[Candidate's Question]: {candidate_question}\n")
    add_to_transcript("user", candidate_question)
    
    closing_message = "Thank you for participating in the interview. We appreciate your time and effort. The interview is now complete."
    print("\n[AI Closing]:", closing_message)
    add_to_transcript("assistant", closing_message)
    speak_text(closing_message)
    
    evaluation = evaluate_candidate(transcript)
    print("\n[AI Candidate Evaluation]:\n", evaluation, "\n")
    add_to_transcript("assistant", evaluation)
    speak_text("Katılımcıyı değerlendiriyorum. " + evaluation)
    
    print("\nFinal Interview Transcript:")
    for msg in transcript["messages"]:
        print(f"{msg['role']}: {msg['content']}\n")

if __name__ == "__main__":
    interview_simulation()
