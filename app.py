"""University Doctor Reviews Assistant - standalone demo.

Run locally:   pip install -r requirements.txt && python app.py
Run online:    create a Hugging Face Space (SDK: Gradio) and upload app.py + requirements.txt
"""

import os
import re
import random

import numpy as np
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel

random.seed(42)


def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = re.sub(r"\s+", " ", text).strip()
    return text


colleges = {
    "computing":   ["البرمجة", "قواعد البيانات", "الذكاء الاصطناعي", "الشبكات"],
    "sciences":    ["الكيمياء", "الفيزياء", "الأحياء", "الرياضيات"],
    "business":    ["المحاسبة", "التسويق", "الإدارة", "الاقتصاد"],
    "engineering": ["الدوائر الكهربائية", "الميكانيكا", "التصميم الهندسي", "الإنشاءات"],
}

dialect_templates = {
    "najdi": {
        "subject": ["الدكتور يشرح {s} حيل زين", "اختبارات {s} صعبة شوي بس عدل", "واجبات {s} كثيرة بس تفيدنا",
                    "ما استفدت شي من محاضرات {s}", "متعاون ويجاوب على اسئلتنا في {s} على طول", "يجي على وقته ويمشي على منهج {s}"],
        "generic": ["الدكتور حيل زين وما قصر معنا", "ما استفدت شي منه الصراحة"],
    },
    "hijazi": {
        "subject": ["الدكتور يشرح {s} حلو قوي", "اختبارات {s} صعبة شوية بس عادلة", "واجبات {s} كتير بس مفيدة",
                    "ما استفدت من محاضرات {s} خالص", "دكتور متعاون ويرد على اسئلتنا في {s} دايما", "دحين فهمت {s} بعد شرحه"],
        "generic": ["الدكتور حلو قوي وبيشرح كويس", "ما استفدت منه خالص"],
    },
    "sharqi": {
        "subject": ["الدكتور يشرح {s} وايد زين", "اختبارات {s} صعبة شوي بس منصفة", "واجبات {s} وايد بس مفيدة",
                    "ما استفدت شي من محاضرات {s} مب زين", "الدكتور متعاون وايد ويجاوب على اسئلتنا في {s}", "ملتزم بوقت المحاضرة ومنهج {s}"],
        "generic": ["الدكتور وايد زين ويفهمنا", "ما استفدت شي منه مب زين"],
    },
    "white": {
        "subject": ["الدكتور يشرح {s} بطريقة واضحة وممتازة", "اختبارات {s} صعبة شوي بس عادلة", "واجبات {s} كثيرة بس مفيدة",
                    "ما استفدت كثير من محاضرات {s}", "الدكتور متعاون ويجاوب على اسئلتنا في {s}", "ملتزم بوقت المحاضرة ومنهج {s}"],
        "generic": ["الدكتور ممتاز ويشرح بشكل جيد", "لم استفد كثيرا من الدكتور"],
    },
}

doctor_names = ["خالد", "فهد", "سعد", "محمد", "عبدالله", "ناصر", "سلطان", "ماجد", "تركي", "بندر"]


def add_typo(message):
    words = message.split()
    candidates = [i for i, w in enumerate(words) if len(w) >= 4]
    if not candidates:
        return message
    i = random.choice(candidates)
    w = words[i]
    pos = random.randrange(1, len(w) - 1)
    if random.random() < 0.5:
        words[i] = w[:pos] + w[pos + 1:]
    else:
        words[i] = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
    return " ".join(words)


def build_dataset():
    if os.path.exists("my_reviews.csv"):
        data = pd.read_csv("my_reviews.csv")
        data["text"] = data["text"].apply(normalize_arabic)
        return data[["text", "college"]].drop_duplicates("text").reset_index(drop=True)

    rows = []
    for college, subjects in colleges.items():
        for _ in range(20):
            subject = random.choice(subjects)
            doctor_name = random.choice(doctor_names)
            for _ in range(6):
                dialect = random.choice(list(dialect_templates))
                if random.random() < 0.2:
                    message = random.choice(dialect_templates[dialect]["generic"])
                else:
                    message = random.choice(dialect_templates[dialect]["subject"]).format(s=subject)
                if random.random() < 0.4:
                    message = "الدكتور " + doctor_name + ": " + message
                if random.random() < 0.3:
                    message = add_typo(message)
                rows.append({"text": normalize_arabic(message), "college": college})
    return pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)


df = build_dataset()
topics = sorted(df["college"].unique())

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
X = vectorizer.fit_transform(df["text"])

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X, df["college"])

similarity_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
S = similarity_vectorizer.fit_transform(df["text"])

all_subjects = [(normalize_arabic(s), college) for college, subs in colleges.items() for s in subs]


def extract_slots(text):
    slots = {}
    m = re.search(r"الدكتور (\S+):", text)
    if m:
        slots["doctor"] = m.group(1)
    for subject, college in all_subjects:
        if subject in text:
            slots["subject"] = subject
            slots["subject_college"] = college
            break
    return slots


def analyze(review, threshold=0.60):
    if not review or not review.strip():
        return {}, "اكتب تقييماً أولاً.", ""

    text = normalize_arabic(review)
    vec = vectorizer.transform([text])
    probs = classifier.predict_proba(vec)[0]
    scores = {c: float(p) for c, p in zip(classifier.classes_, probs)}

    best = classifier.classes_[int(probs.argmax())]
    confidence = float(probs.max())
    slots = extract_slots(text)

    if confidence >= threshold:
        verdict = "الكلية: " + best + " (من النموذج، الثقة " + str(round(confidence, 2)) + ")"
    elif "subject_college" in slots:
        verdict = "الكلية: " + slots["subject_college"] + " (من اسم المادة، ثقة النموذج منخفضة " + str(round(confidence, 2)) + ")"
    else:
        verdict = "غير واضح: التقييم لا يكفي لتحديد الكلية (الثقة " + str(round(confidence, 2)) + ")"

    details = verdict
    if "doctor" in slots:
        details += "\nالدكتور: " + slots["doctor"]
    if "subject" in slots:
        details += "\nالمادة: " + slots["subject"]

    similarity = linear_kernel(similarity_vectorizer.transform([text]), S).ravel()
    order = [i for i in similarity.argsort()[::-1] if df["text"].iloc[i] != text][:3]
    similar = "\n".join("- " + df["text"].iloc[i] for i in order)

    return scores, details, similar


examples = [
    "الدكتور خالد: اختبارات الكيمياء صعبة شوي بس عدل",
    "ما استفدت شي من محاضرات قواعد البيانات",
    "دكتور متعاون ويرد على اسئلتنا في الاقتصاد دايما",
    "الدكتور حلو قوي وبيشرح كويس",
]

with gr.Blocks(title="University Doctor Reviews Assistant") as demo:
    gr.Markdown("# مساعد تقييمات الدكاترة الجامعيين\nUniversity Doctor Reviews Assistant\n\nاكتب تقييم طالب (بأي لهجة سعودية) وسيحدد الكلية ويلاقي تقييمات مشابهة.")
    review_box = gr.Textbox(label="تقييم الطالب", lines=3, placeholder="مثال: اختبارات الكيمياء صعبة شوي بس عدل")
    run_button = gr.Button("حلّل التقييم", variant="primary")
    scores_box = gr.Label(label="احتمال كل كلية", num_top_classes=4)
    details_box = gr.Textbox(label="النتيجة", lines=3)
    similar_box = gr.Textbox(label="تقييمات مشابهة", lines=4)
    gr.Examples(examples=examples, inputs=review_box)
    run_button.click(analyze, inputs=review_box, outputs=[scores_box, details_box, similar_box], api_name="analyze")

if __name__ == "__main__":
    demo.launch()
