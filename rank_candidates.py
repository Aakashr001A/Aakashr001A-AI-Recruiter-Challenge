import json
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------
# Load Candidates
# --------------------

candidates = []

with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidates.append(json.loads(line))

print(f"Loaded {len(candidates)} candidates")

# --------------------
# Load Job Description
# --------------------

doc = Document("job_description.docx")

job_description = ""

for para in doc.paragraphs:
    job_description += para.text + "\n"

print("\nJOB DESCRIPTION PREVIEW:")
print(job_description[:1000])

# --------------------
# Load Embedding Model
# --------------------

print("\nLoading AI Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

job_embedding = model.encode(job_description)

print("Model Loaded Successfully!")

# --------------------
# Important Skills
# --------------------

important_skills = [
    "Python",
    "FAISS",
    "Pinecone",
    "Milvus",
    "Weaviate",
    "Qdrant",
    "Elasticsearch",
    "OpenSearch",
    "Embeddings",
    "Retrieval",
    "Ranking",
    "NDCG",
    "MAP",
    "MRR",
    "LLM",
    "Vector Database"
]

# --------------------
# Skill Score
# --------------------

def calculate_skill_score(candidate):

    candidate_skills = [
        skill["name"].lower()
        for skill in candidate.get("skills", [])
    ]

    score = 0

    for skill in important_skills:
        if skill.lower() in candidate_skills:
            score += 1

    return score

# --------------------
# Experience Score
# --------------------

def calculate_experience_score(candidate):

    exp = candidate["profile"]["years_of_experience"]

    if 5 <= exp <= 9:
        return 10
    elif 3 <= exp < 5:
        return 7
    elif 9 < exp <= 12:
        return 6
    else:
        return 2

# --------------------
# Behavioral Score
# --------------------

def calculate_behavior_score(candidate):

    signals = candidate.get("redrob_signals", {})

    response_rate = signals.get("recruiter_response_rate", 0)
    interview_rate = signals.get("interview_completion_rate", 0)

    github_score = signals.get("github_activity_score", 0)

    if github_score == -1:
        github_score = 0

    open_to_work = 1 if signals.get("open_to_work_flag", False) else 0

    saves = signals.get("saved_by_recruiters_30d", 0)

    score = (
        response_rate * 4 +
        interview_rate * 3 +
        open_to_work * 2 +
        github_score / 50 +
        saves / 10
    )

    return round(score, 2)

# --------------------
# Semantic Score
# --------------------

def calculate_semantic_score(candidate):

    headline = candidate["profile"].get("headline", "")

    skills = " ".join(
        [skill["name"] for skill in candidate.get("skills", [])]
    )

    candidate_text = headline + " " + skills

    candidate_embedding = model.encode(candidate_text)

    similarity = cosine_similarity(
        [job_embedding],
        [candidate_embedding]
    )[0][0]

    return round(similarity * 10, 2)

# --------------------
# AI Bonus
# --------------------

def calculate_ai_bonus(candidate):

    headline = candidate["profile"].get("headline", "").lower()

    ai_keywords = [
    "ai engineer",
    "machine learning",
    "ml engineer",
    "nlp",
    "recommendation",
    "retrieval",
    "ranking",
    "search",
    "data scientist",
    "applied ml",
    "llm",
    "genai",
    "rag",
    "langchain",
    "deep learning",
    "vector database",
    "ai research"
]

    score = 0

    for keyword in ai_keywords:
        if keyword in headline:
            score += 2

    negative_roles = [
        "graphic designer",
        "marketing manager",
        "accountant",
        "hr manager",
        "content writer",
        "civil engineer",
        "mechanical engineer",
        "operations manager",
        "project manager",
        "business analyst",
    ]

    for role in negative_roles:
        if role in headline:
            score -= 10

    return score

    

# --------------------
# Final Score
# --------------------

def calculate_final_score(candidate):

    semantic_score = calculate_semantic_score(candidate)

    skill_score = calculate_skill_score(candidate)

    experience_score = calculate_experience_score(candidate)

    behavior_score = calculate_behavior_score(candidate)

    ai_bonus = calculate_ai_bonus(candidate)

    final_score = (
        semantic_score * 6 +
        skill_score * 4 +
        experience_score * 2 +
        behavior_score +
        ai_bonus
    )

    return round(final_score, 2)
# --------------------
# Rank Candidates
# --------------------

ranked_candidates = []

for candidate in candidates[:1000]:

    final_score = calculate_final_score(candidate)

    ranked_candidates.append({
        "candidate_id": candidate["candidate_id"],
        "headline": candidate["profile"]["headline"],
        "experience": candidate["profile"]["years_of_experience"],
        "score": final_score
    })

# Sort by score (highest first)
ranked_candidates.sort(
    key=lambda x: x["score"],
    reverse=True
)

import pandas as pd

submission_rows = []

for rank, candidate in enumerate(ranked_candidates, start=1):

    reasoning = (
        f"{candidate['headline']} "
        f"with {candidate['experience']} years experience"
    )

    submission_rows.append({
        "candidate_id": candidate["candidate_id"],
        "rank": rank,
        "score": round(candidate["score"] / 100, 3),
        "reasoning": reasoning
    })

submission = pd.DataFrame(submission_rows)

submission.to_csv(
    "ranked_output.csv",
    index=False
)

print("\nSubmission file saved as ranked_output.csv")
# --------------------
# Show Top 20
# --------------------

print("\nTOP 20 CANDIDATES:\n")

for i, candidate in enumerate(ranked_candidates[:20], start=1):

    print(
        f"{i}.",
        candidate["candidate_id"],
        "|",
        candidate["headline"],
        "| Score:",
        candidate["score"]
    )