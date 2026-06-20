import os
import json
import docx
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

app = FastAPI(title="AI Recruiter API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Global Data Cache
# ----------------------------------------
JOB_DETAILS = {
    "title": "Senior AI / Search & Recommendation Engineer",
    "description": ""
}
SAMPLE_CANDIDATES = []
RANKED_CSV_DATA = []
CANDIDATE_SUB_SCORES = {} # candidate_id -> dict of sub-scores
MODEL = None
JOB_EMBEDDING = None

# Important Skills from rank_candidates.py
IMPORTANT_SKILLS = [
    "Python", "FAISS", "Pinecone", "Milvus", "Weaviate", "Qdrant",
    "Elasticsearch", "OpenSearch", "Embeddings", "Retrieval",
    "Ranking", "NDCG", "MAP", "MRR", "LLM", "Vector Database"
]

AI_KEYWORDS = [
    "ai engineer", "machine learning", "ml engineer", "nlp",
    "recommendation", "retrieval", "ranking", "search",
    "data scientist", "applied ml", "llm", "genai", "rag",
    "langchain", "deep learning", "vector database", "ai research"
]

NEGATIVE_ROLES = [
    "graphic designer", "marketing manager", "accountant",
    "hr manager", "content writer", "civil engineer",
    "mechanical engineer", "operations manager", "project manager",
    "business analyst"
]

# ----------------------------------------
# Loading Helpers
# ----------------------------------------
def load_job_description():
    global JOB_DETAILS
    try:
        if os.path.exists("job_description.docx"):
            doc = docx.Document("job_description.docx")
            text = ""
            for p in doc.paragraphs:
                text += p.text + "\n"
            JOB_DETAILS["description"] = text.strip()
        else:
            JOB_DETAILS["description"] = "Senior Software Engineer with focus on Python, Vector Databases, LLMs, and Retrieval-Augmented Generation."
    except Exception as e:
        print(f"Error loading job description docx: {e}")
        JOB_DETAILS["description"] = "Senior Software Engineer with focus on Python, Vector Databases, LLMs, and Retrieval-Augmented Generation."

def precompute_candidate_scores():
    global SAMPLE_CANDIDATES, CANDIDATE_SUB_SCORES, MODEL, JOB_EMBEDDING
    
    # Load SentenceTransformer
    print("Loading SentenceTransformer model...")
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    job_desc = JOB_DETAILS["description"]
    JOB_EMBEDDING = MODEL.encode(job_desc)
    print("AI Model loaded and job description encoded.")
    
    # Load sample candidates json
    if os.path.exists("sample_candidates.json"):
        with open("sample_candidates.json", "r", encoding="utf-8") as f:
            SAMPLE_CANDIDATES = json.load(f)
        print(f"Loaded {len(SAMPLE_CANDIDATES)} sample candidates.")
    else:
        print("sample_candidates.json not found!")
        SAMPLE_CANDIDATES = []

    # Precompute scores for the 50 candidates
    for candidate in SAMPLE_CANDIDATES:
        cid = candidate["candidate_id"]
        
        # 1. Skill Score
        candidate_skills = [s["name"].lower() for s in candidate.get("skills", [])]
        skill_score = sum(1 for skill in IMPORTANT_SKILLS if skill.lower() in candidate_skills)
        
        # 2. Experience Score
        exp = candidate["profile"]["years_of_experience"]
        if 5 <= exp <= 9:
            exp_score = 10
        elif 3 <= exp < 5:
            exp_score = 7
        elif 9 < exp <= 12:
            exp_score = 6
        else:
            exp_score = 2
            
        # 3. Behavioral Score
        signals = candidate.get("redrob_signals", {})
        resp_rate = signals.get("recruiter_response_rate", 0)
        int_rate = signals.get("interview_completion_rate", 0)
        gh_score = signals.get("github_activity_score", 0)
        if gh_score == -1:
            gh_score = 0
        open_to_work = 1 if signals.get("open_to_work_flag", False) else 0
        saves = signals.get("saved_by_recruiters_30d", 0)
        behavior_score = round(resp_rate * 4 + int_rate * 3 + open_to_work * 2 + gh_score / 50 + saves / 10, 2)
        
        # 4. Semantic Score
        headline = candidate["profile"].get("headline", "")
        skills_str = " ".join([s["name"] for s in candidate.get("skills", [])])
        cand_text = headline + " " + skills_str
        cand_emb = MODEL.encode(cand_text)
        sim = cosine_similarity([JOB_EMBEDDING], [cand_emb])[0][0]
        semantic_score = round(sim * 10, 2)
        
        # 5. AI Bonus
        headline_lower = headline.lower()
        ai_bonus = sum(2 for kw in AI_KEYWORDS if kw in headline_lower)
        negative_score = sum(-10 for role in NEGATIVE_ROLES if role in headline_lower)
        ai_bonus += negative_score
        
        # Store in dict
        CANDIDATE_SUB_SCORES[cid] = {
            "semantic": semantic_score,
            "skills": skill_score,
            "experience": exp_score,
            "behavior": behavior_score,
            "ai_bonus": ai_bonus,
            "skills_list": [s["name"] for s in candidate.get("skills", [])],
            "years_of_experience": exp,
            "headline": headline,
            "name": candidate["profile"].get("anonymized_name", f"Candidate {cid[-5:]}")
        }
    print("Precomputation finished.")

# Load the ranked output CSV
def load_ranked_csv():
    global RANKED_CSV_DATA
    if os.path.exists("ranked_output.csv"):
        df = pd.read_csv("ranked_output.csv")
        RANKED_CSV_DATA = []
        for _, row in df.iterrows():
            RANKED_CSV_DATA.append({
                "candidate_id": str(row["candidate_id"]),
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "reasoning": str(row["reasoning"])
            })
        print(f"Loaded {len(RANKED_CSV_DATA)} ranked outputs from CSV.")
    else:
        print("ranked_output.csv not found!")
        RANKED_CSV_DATA = []

@app.on_event("startup")
def startup_event():
    load_job_description()
    load_ranked_csv()
    precompute_candidate_scores()

# ----------------------------------------
# Endpoints
# ----------------------------------------

@app.get("/api/jobs")
def get_jobs():
    return {
        "job_id": "JOB_001",
        "title": JOB_DETAILS["title"],
        "description": JOB_DETAILS["description"],
        "important_skills": IMPORTANT_SKILLS
    }

@app.get("/api/candidates")
def get_candidates(
    w_semantic: float = Query(6.0, description="Semantic Match Weight"),
    w_skills: float = Query(4.0, description="Skills Match Weight"),
    w_experience: float = Query(2.0, description="Experience Relevance Weight"),
    w_behavior: float = Query(1.0, description="Behavioral Signals Weight"),
    search: Optional[str] = Query(None, description="Search by name, headline, or skill")
):
    results = []
    
    # Use precomputed sample candidates
    for cid, scores in CANDIDATE_SUB_SCORES.items():
        # Recalculate dynamic overall score
        raw_score = (
            scores["semantic"] * w_semantic +
            scores["skills"] * w_skills +
            scores["experience"] * w_experience +
            scores["behavior"] * w_behavior +
            scores["ai_bonus"]
        )
        # Normalize score to be out of 1.0 (approximate)
        final_score = round(raw_score / 100, 3)
        
        # Determine status fit
        if final_score >= 0.55:
            fit_status = "Fit"
        elif final_score >= 0.45:
            fit_status = "Shortlist"
        elif final_score >= 0.30:
            fit_status = "Review"
        else:
            fit_status = "Reject"
            
        candidate_data = {
            "candidate_id": cid,
            "name": scores["name"],
            "headline": scores["headline"],
            "experience": scores["years_of_experience"],
            "score": final_score,
            "fit_status": fit_status,
            "sub_scores": {
                "semantic": scores["semantic"],
                "skills": scores["skills"],
                "experience": scores["experience"],
                "behavior": scores["behavior"],
                "ai_bonus": scores["ai_bonus"]
            },
            "skills": scores["skills_list"][:5] # Return top 5 skills
        }
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            if (search_lower in scores["name"].lower() or
                search_lower in scores["headline"].lower() or
                any(search_lower in sk.lower() for sk in scores["skills_list"])):
                results.append(candidate_data)
        else:
            results.append(candidate_data)
            
    # Sort results by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Assign ranks dynamically based on current sorting
    for rank, cand in enumerate(results, start=1):
        cand["rank"] = rank
        
    return results

@app.get("/api/candidates/{candidate_id}")
def get_candidate_detail(candidate_id: str):
    # Find full details from SAMPLE_CANDIDATES
    full_cand = next((c for c in SAMPLE_CANDIDATES if c["candidate_id"] == candidate_id), None)
    
    if not full_cand:
        # Check if they are in the CSV data but not the sample candidates
        csv_cand = next((c for c in RANKED_CSV_DATA if c["candidate_id"] == candidate_id), None)
        if csv_cand:
            # Return basic metadata
            return {
                "candidate_id": candidate_id,
                "profile": {
                    "anonymized_name": f"Candidate {candidate_id[-5:]}",
                    "headline": csv_cand["reasoning"],
                    "summary": f"Ranked #{csv_cand['rank']} in overall candidate database with a base score of {csv_cand['score']}.",
                    "years_of_experience": 5.0
                },
                "skills": [],
                "education": [],
                "career_history": [],
                "redrob_signals": {},
                "score": csv_cand["score"],
                "strengths": ["Strong overall rank match in candidate pool"],
                "gaps": ["Detailed profile information was not included in sample dataset"]
            }
        return {"error": "Candidate not found"}
        
    # Compile explainability (strengths and gaps)
    scores = CANDIDATE_SUB_SCORES.get(candidate_id, {})
    strengths = []
    gaps = []
    
    # Evaluate semantic match
    if scores.get("semantic", 0) >= 6.5:
        strengths.append(f"Excellent semantic alignment with job description (similarity score: {scores['semantic']}/10)")
    elif scores.get("semantic", 0) < 4.0:
        gaps.append("Weak alignment between profile resume summary and target job description requirements")
        
    # Evaluate skills
    matching_skills = [sk for sk in IMPORTANT_SKILLS if sk.lower() in [s["name"].lower() for s in full_cand.get("skills", [])]]
    missing_skills = [sk for sk in IMPORTANT_SKILLS if sk.lower() not in [s["name"].lower() for s in full_cand.get("skills", [])]]
    
    if len(matching_skills) >= 5:
        strengths.append(f"Highly qualified in critical tech stack: {', '.join(matching_skills[:5])}")
    elif len(matching_skills) > 0:
        strengths.append(f"Possesses core skills: {', '.join(matching_skills)}")
    
    if len(missing_skills) > 5:
        gaps.append(f"Missing core job technologies: {', '.join(missing_skills[:5])}")
        
    # Evaluate experience
    exp_years = full_cand["profile"]["years_of_experience"]
    if 5 <= exp_years <= 9:
        strengths.append(f"Optimal professional experience level ({exp_years} years)")
    elif exp_years > 12:
        strengths.append(f"Significant industry tenure ({exp_years} years), potentially overqualified for a mid-to-senior role")
    elif exp_years < 3:
        gaps.append(f"Short overall professional experience ({exp_years} years) compared to the requested senior-level profile")
        
    # Evaluate signals
    signals = full_cand.get("redrob_signals", {})
    if signals.get("open_to_work_flag"):
        strengths.append("Actively seeking new opportunities (Open to Work flag active)")
    if signals.get("recruiter_response_rate", 0) > 0.8:
        strengths.append("Highly responsive to recruiter messages")
    if signals.get("github_activity_score", 0) > 40:
        strengths.append(f"Strong open-source engagement (GitHub activity score: {signals['github_activity_score']})")
    if signals.get("interview_completion_rate", 0) < 0.4 and signals.get("interview_completion_rate", 0) > 0:
        gaps.append("High historical interview drop-off rate (low completion rate)")
        
    return {
        "candidate_id": candidate_id,
        "profile": full_cand["profile"],
        "skills": full_cand.get("skills", []),
        "education": full_cand.get("education", []),
        "career_history": full_cand.get("career_history", []),
        "redrob_signals": signals,
        "score": scores.get("semantic", 0) * 0.06 + scores.get("skills", 0) * 0.04 + scores.get("experience", 0) * 0.02 + scores.get("behavior", 0) * 0.01, # placeholder default
        "sub_scores": {
            "semantic": scores.get("semantic"),
            "skills": scores.get("skills"),
            "experience": scores.get("experience"),
            "behavior": scores.get("behavior"),
            "ai_bonus": scores.get("ai_bonus")
        },
        "strengths": strengths,
        "gaps": gaps
    }

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    # Simulates parsing a new batch of candidate profiles and ranking them
    try:
        contents = await file.read()
        import io
        import random
        # Parse simple CSV data
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        
        # Generate mock rankings for the uploaded CSV candidates
        uploaded_candidates = []
        for i, row in df.iterrows():
            cid = row.get("candidate_id")
            if pd.isna(cid) or not cid:
                cid = f"CAND_UP_{1000 + i}"
            else:
                cid = str(cid)

            name = row.get("name")
            if pd.isna(name) or not name:
                name = f"Candidate {1000 + i}"
            else:
                name = str(name)

            headline = row.get("headline")
            if pd.isna(headline) or not headline:
                headline = row.get("reasoning")
                if pd.isna(headline) or not headline:
                    headline = "Software Engineer"
                else:
                    headline = str(headline)
            else:
                headline = str(headline)

            experience_val = row.get("experience")
            if pd.isna(experience_val) or experience_val is None:
                experience_val = row.get("years_of_experience")
            
            try:
                experience = float(experience_val) if not pd.isna(experience_val) and experience_val is not None else 4.0
            except:
                experience = 4.0

            # Score
            score_val = row.get("score")
            try:
                score = float(score_val) if not pd.isna(score_val) and score_val is not None else round(0.4 + random.random() * 0.45, 3)
            except:
                score = round(0.4 + random.random() * 0.45, 3)
            
            fit_status = "Fit" if score >= 0.65 else ("Shortlist" if score >= 0.5 else ("Review" if score >= 0.35 else "Reject"))
            
            uploaded_candidates.append({
                "candidate_id": cid,
                "name": name,
                "headline": headline,
                "experience": float(experience),
                "score": float(score),
                "fit_status": fit_status,
                "skills": ["Python", "SQL", "Docker"],
                "sub_scores": {
                    "semantic": float(round(score * 10, 1)),
                    "skills": 4,
                    "experience": 6,
                    "behavior": 5,
                    "ai_bonus": 2
                }
            })
            
        uploaded_candidates.sort(key=lambda x: x["score"], reverse=True)
        for rank, cand in enumerate(uploaded_candidates, start=1):
            cand["rank"] = int(rank)
            
        return {
            "message": f"Successfully parsed and ranked {len(uploaded_candidates)} candidates.",
            "candidates": uploaded_candidates
        }
    except Exception as e:
        return {"error": f"Failed to process CSV file: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
