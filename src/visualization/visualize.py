# visualization.py

import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

def generate_visualizations(data):
    try:
        # Example visualization: Scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(data['x'], data['y'], c='blue')
        plt.title('Scatter Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # Convert plot to PNG image
        png_image = io.BytesIO()
        plt.savefig(png_image, format='png')
        png_image.seek(0)
        png_image_base64 = base64.b64encode(png_image.read()).decode('utf8')

        plt.close()
        return png_image_base64
    except Exception as e:
        raise RuntimeError(f"Error generating visualizations: {str(e)}")



def create_stock_chart(stock_prices_df, predicted_prices, symbol):
	plt.figure(figsize=(12, 6))
	plt.plot(stock_prices_df.index, stock_prices_df['1. open'], label='Historical Prices')
	future_dates = pd.date_range(start=stock_prices_df.index[-1] + pd.Timedelta(days=1), periods=len(predicted_prices))
	plt.plot(future_dates, predicted_prices, label='Predicted Prices', color='red')
	plt.title(f'{symbol} Stock Price Prediction')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()

	buffer = io.BytesIO()
	plt.savefig(buffer, format='png')
	buffer.seek(0)
	image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
	plt.close()

	return image_base64
