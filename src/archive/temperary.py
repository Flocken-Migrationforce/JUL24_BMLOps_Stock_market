# run_train.py
import matplotlib.pyplot as plt 

# Predict Google (GOOGL) stock prices
predictions_val, actual_values, next_week_predictions = train_validate_predict('GTLB', start_date='2020-01-01', end_date='2024-08-014', interval='1d')

# Print the validation predictions and the actual values
print("Validation Predictions vs Actual Values:")
for pred, actual in zip(predictions_val, actual_values):
    print(f"Predicted: {pred[0]:.2f}, Actual: {actual[0]:.2f}")

plt.plot(predictions_val)
plt.plot(actual_values)
plt.show()



# Print the predicted prices for the next week
print("Predicted prices for the next week:")
for i, price in enumerate(next_week_predictions, start=1):
    print(f"Day {i}: {price[0]:.2f}")
