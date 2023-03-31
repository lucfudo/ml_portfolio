# Anime Recommendation API
This API provides top N anime recommendations based on user ratings. It is built with *FastAPI* and uses a collaborative filtering algorithm to generate recommendations.

Request
The request must be a POST request with a JSON body containing the following fields:

user_id (required): The ID of the user to generate recommendations for.

Example request body:

```json
{
  "user_id": 1,
}
```
Response
The response will be a JSON object containing the following fields:

user_id: The ID of the user the recommendations are for.
top_n: An array of the top N anime recommendations.
Example response:

```json
{
  "top_n": [
    [
      1234,
      "Naruto",
      8.5
    ],
    [
      5678,
      "One Piece",
      10
    ],
    ...
  ]
}
```
