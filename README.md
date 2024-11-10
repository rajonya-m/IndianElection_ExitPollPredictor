# Election-Exit-Poll-Prediction
 The Exit Poll Predictor aims to revolutionize exit polling methods by integrating two powerful techniques—text-based sentiment analysis and real-time emotion tracking to provide deeper insights into voter preferences. Traditional exit polls have limitations: they often rely solely on verbal responses, which can be incomplete, biased, or influenced by social pressures. The addition of emotion tracking through facial analysis offers a more authentic measure, capturing both verbalized sentiments and underlying emotions. The project focuses on Indian political parties, specifically BJP and Congress, assessing voter sentiment through spoken responses and emotion scores tracked via webcam during interviews. We developed a Real Emotion Score formula combining sentiment analysis of text (verbal response) and real-time emotion scores (facial expressions), providing a holistic analysis of each voter’s preference.

Sentiment_Score: Using sentiment analysis from voters’ text responses during the interview.
Emotion_Score: Calculated by outputting the average of continuous emotion scores recorded by a webcam during the interview of visual emotion on the voter’s face while answering.
Real_Emotion : alpha * Sentiment_Score + beta * Emotion_Score
The final output shows the true, more accurate exit poll preference of a political party based on the overall emotion and response of every voter.

Libraries: NLTK and Vader for sentiment, OpenCV for facial recognition, and Matplotlib/Seaborn for visualization.
 
