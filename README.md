# **Alumni-Student Matching Recommender System**

## **Overview**

This project is a **recommender system** designed to connect students with alumni of their institution who share similar interests. By analyzing LinkedIn profiles, the system identifies commonalities such as skills, work experience, and interests to facilitate meaningful connections between students and alumni.

---

## **Features**

- Matches students with alumni based on shared interests, skills, and professional goals.
- Utilizes LinkedIn profile data for accurate recommendations.
- Provides a user-friendly interface for students and alumni.
- Enhances networking opportunities and mentorship possibilities.

---

## **How It Works**

1. **Data Collection:**
   - The system extracts data from LinkedIn profiles of students and alumni, focusing on:
     - Skills
     - Work experience
     - Interests
     - Education
2. **Data Preprocessing:**
   - Cleans and standardizes the profile data.
   - Extracts keywords and calculates similarity metrics.
3. **Recommendation Engine:**
   - Uses techniques like **Cosine Similarity**, **TF-IDF**, or **Word Embeddings** to find matches.
   - Generates a ranked list of alumni for each student based on their profile similarity.
4. **Output:**
   - Displays a list of recommended alumni with details like their name, designation, and shared interests.

---

## **Technologies Used**

- **Backend:**
  - Python, Flask
  - Natural Language Processing (NLP) libraries
- **Machine Learning:**
  - Scikit-learn
- **APIs:**
  - Scrupp API for profile data extraction.

---

