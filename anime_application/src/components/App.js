import React from "react";
import './index.css';

const API = "http://localhost:8001/predict"

const App = () => {
    const [isLoading, setIsLoading] = React.useState(false);
    const [userId, setUserId] = React.useState('');
    const [topN, setTopN] = React.useState([]);
  
    const handleSubmit = async (event) => {
      event.preventDefault();
      setIsLoading(true);
  
      try {
        const response = await fetch(API, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ user_id: userId })
        });
        const data = await response.json();
        setTopN(data.top_n);
        setIsLoading(false)
      } catch (error) {
        console.error(error);
        setTopN([]);
      }
    };
  
    return (
      <div className="container">
        <h1>Anime Recommendation App</h1>
        <form onSubmit={handleSubmit}>
          <label>
            <input
              type="number"
              value={userId}
              placeholder="User ID:"
              onChange={(event) => setUserId(event.target.value)}
            />
          </label>
          <button type="submit">Get Recommendations</button>
        </form>
        <ul>
          {isLoading && <div className="loading">
              <div className="spinner"></div>
            </div>
          }
          {topN && topN.map(([title, score, malId]) => (
            <li key={malId}>
              <div className="anime-card">
                <img
                  src={`https://source.unsplash.com/300x400/?anime,${title}`}
                  alt={title}
                />
                <div className="anime-details">
                  <b>{title}</b>
                  <p>Score: {score.toFixed(2)}</p>
                </div>
              </div>
            </li>
          ))}
        </ul>
      </div>
    );
}

export default App;
