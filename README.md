# AI Recruiter – Intelligent Candidate Ranking System

## Overview

AI Recruiter is an AI-powered candidate ranking system designed to identify the best-fit candidates for a given job description. The system combines semantic matching, skill analysis, experience evaluation, and behavioral signals to generate an intelligent ranked shortlist.

Traditional Applicant Tracking Systems (ATS) often rely heavily on keyword matching, which can overlook qualified candidates. This project uses Natural Language Processing (NLP) and semantic embeddings to understand the context and meaning of candidate profiles and job descriptions.

---

## Problem Statement

Recruiters often receive thousands of applications for a single role. Manually reviewing candidates is time-consuming and inefficient.

The objective of this project is to:

* Understand complex job descriptions.
* Identify relevant candidate skills and experience.
* Analyze recruiter engagement and behavioral signals.
* Generate a ranked shortlist of candidates.
* Improve candidate-job matching beyond simple keyword search.

---

## Features

### Semantic Candidate Matching

Uses Sentence Transformers to understand the meaning and context of job descriptions and candidate profiles.

### Skill-Based Ranking

Evaluates candidates against important technical skills required for the role.

### Experience Scoring

Prioritizes candidates whose experience aligns with job requirements.

### Behavioral Signal Analysis

Uses recruiter response rates, interview completion rates, GitHub activity, and recruiter saves to improve ranking quality.

### Intelligent Candidate Ranking

Combines multiple scoring factors into a single ranking score.

---

## Technology Stack

* Python
* Sentence Transformers
* Scikit-learn
* Pandas
* Python-Docx

### AI Model

* all-MiniLM-L6-v2

---

## System Architecture

Job Description

↓

Semantic Matching

↓

Skill Analysis

↓

Experience Analysis

↓

Behavioral Signal Analysis

↓

Final Candidate Ranking

↓

Ranked Candidate Shortlist

---

## Ranking Methodology

The final candidate score is calculated using:

### Semantic Score

Measures similarity between the job description and candidate profile using sentence embeddings and cosine similarity.

### Skill Score

Evaluates candidate skills against role-specific requirements.

### Experience Score

Rewards candidates whose experience matches the desired experience range.

### Behavioral Score

Uses:

* Recruiter Response Rate
* Interview Completion Rate
* GitHub Activity Score
* Open To Work Status
* Recruiter Saves

### AI Relevance Bonus

Provides additional weight to candidates with AI, Machine Learning, Retrieval, Ranking, Search, and LLM-related expertise.

---

## Input Files

* candidates.jsonl – Contains candidate profiles, skills, experience, and behavioral signals.
* job_description.docx – Contains the job description used for candidate matching and ranking.

---

## Output

The system generates:

* ranked_output.csv – Ranked shortlist of candidates with scores and reasoning.

Output columns:

* candidate_id
* rank
* score
* reasoning

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python rank_candidates.py
```

Generated output:

```bash
ranked_output.csv
```

---

## Results

The system successfully:

* Processes candidate profiles.
* Understands job descriptions semantically.
* Evaluates skills and experience.
* Incorporates behavioral signals.
* Produces a ranked candidate shortlist.

---

## Note on Dataset Processing

The provided challenge dataset contains 100,000 candidate profiles.

For local testing and demonstration purposes, the submitted ranked_output.csv was generated using a subset of 1,000 candidates due to computational constraints associated with semantic embedding generation.

The ranking pipeline is designed to support larger datasets and can be scaled to process the complete candidate pool with additional compute resources and optimization techniques.





