
import pandas as pd
from textblob import TextBlob


# Define the sentiment analysis function with error handling
def get_sentiment(text):
    if isinstance(text, str):
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"
    else:
        return "Neutral"  # Default to neutral if the text is not a string


def main():
    # Load the dataset
    file_path = '/Users/rohan/Downloads/archive/test.csv'  # Update with your dataset's path
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Inspect the dataset structure
    print("Dataset Preview:")
    print(df.head())

    # Apply sentiment analysis
    df['Sentiment'] = df['text'].apply(get_sentiment)

    # Display results with the sentiment column
    print("\nSentiment Analysis Results:")
    print(df[['text', 'Sentiment']].head())

    # Save the results to a new CSV file
    output_file = 'sentiment_analysis_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Display sentiment distribution
    print("\nSentiment Distribution:")
    print(df['Sentiment'].value_counts())

if __name__ == "__main__":
    main()