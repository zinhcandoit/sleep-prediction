# sleep-prediction

## 1. Features
- Display your heart rate and counter since pressing the button
- Tracking your sleep quality
- Alarm set for waking up
- Small and easy to use
## 2. Hardware
- **ESP32-C3 SuperMini**: Microcontroller
- **MPU6050**: Accelerometer
- **MAX30102**: Heart Pulse Sensor
- **Buzzer**: Wake up sound / Confirm tracking process
- **Button**: Start/Stop tracking your sleep
## 3. Tech stack
- **Hardware**: ESP32 + MQTT
- **Database**: mongodb
- **FE+BE**: NodeRed
- **Models**: CNN+LSTM for sleep stage prediction with tabular data, Polynomial Regression with n = 2 and ElasticNet for sleep quality prediction 
## 4. Data source
- **Sleep Quality Dataset** from kaggle: Abdullah, Y. (2025). Smartwatch Sleep Tracking Dataset (Synthetic, 2018–2025). Synthetic Dataset for AI Research and Education.
- **Sleep Stage Dataset**: Olivia Walch, Yitong Huang, Daniel Forger, Cathy Goldstein, Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device, Sleep, Volume 42, Issue 12, December 2019, zsz180, https://doi.org/10.1093/sleep/zsz180  