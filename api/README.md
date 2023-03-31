# Anime Recommendation API
This API provides top N anime recommendations based on user ratings. It is built with *FastAPI* and uses a collaborative filtering algorithm to generate recommendations.

Requirements
Python 3.8 or higher
FastAPI
Surprise
Pandas
Installation
Clone the repository: git clone https://github.com/your_username/anime-recommendation-api.git
Install the required dependencies: pip install -r requirements.txt
Start the server: uvicorn main:app --reload
Usage
The API provides a single endpoint to get top N anime recommendations for a user.

Endpoint
/recommendations

Request
The request must be a POST request with a JSON body containing the following fields:

user_id (required): The ID of the user to generate recommendations for.
n (optional): The number of recommendations to generate. Default is 10.
Example request body:

```json
Copy code
{
  "user_id": 1,
  "n": 5
}
Response
The response will be a JSON object containing the following fields:

user_id: The ID of the user the recommendations are for.
recommendations: An array of the top N anime recommendations.
Example response:

json
Copy code
{
  "user_id": 1,
  "recommendations": [
    {
      "anime_id": 1234,
      "anime_title": "Naruto",
      "score": 8.5
    },
    {
      "anime_id": 5678,
      "anime_title": "One Piece",
      "score": 10
    },
    ...
  ]
}
```
