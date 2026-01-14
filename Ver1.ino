#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_MPU6050.h>
#include <PubSubClient.h>

#include "MAX30105.h"
#include "heartRate.h"

// Customize
const char *ssid = "your_internet";
const char *password = "your_password";
const char* mqtt_server = "your_host"; //Your MQTT server. For example: 10.56.15.175
const int mqtt_port = 1883; // MQTT Port

// MQTT topic
const char* TOPIC_SLEEP_DATA   = "sleep/data";
const char* TOPIC_SLEEP_BUTTON = "sleep/button";
const char* TOPIC_BUZZER       = "sleep/buzzer";

#define I2C_SDA     8
#define I2C_SCL     9
#define BUZZER_PIN  3
#define BUTTON_PIN  10

Adafruit_SSD1306 display(128, 64, &Wire, -1);
Adafruit_MPU6050 mpu;
MAX30105 particleSensor;
WiFiClient espClient;
PubSubClient client(espClient);

// HR
const byte RATE_SIZE = 3;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
int hr = 0;

// MPU
float sum_ax = 0, sum_ay = 0, sum_az = 0;
int accCount = 0;

// Timer
unsigned long lastOLEDMillis = 0;
unsigned long lastPublishMillis = 0;
unsigned long sleepStartMillis = 0;

const unsigned long OLED_INTERVAL    = 1000;
const unsigned long PUBLISH_INTERVAL = 30000;

//State
bool isSleepSessionActive = false;

//Set Subcribe for buzzer
// MQTT Callback
void callback(char* topic, byte* payload, unsigned int length) {
  if (String(topic) == TOPIC_BUZZER && length > 0) {
    if (payload[0] == '1' || payload[0] == 1) {
      digitalWrite(BUZZER_PIN, HIGH);
    } else {
      digitalWrite(BUZZER_PIN, LOW);
    }
  }
}

//MQTT Connect
void reconnect() {
  static unsigned long lastTry = 0;
  if (millis() - lastTry < 5000) return;
  lastTry = millis();

  if (client.connect("ESP32-C3")) {
    client.subscribe(TOPIC_BUZZER, 1);
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  pinMode(BUTTON_PIN, INPUT);  

  Wire.begin(I2C_SDA, I2C_SCL);

  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.setTextColor(WHITE);

  mpu.begin();
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);

  particleSensor.begin(Wire, I2C_SPEED_FAST);
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x3F);
  
  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);
  WiFi.setTxPower(WIFI_POWER_11dBm);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  Serial.println(WiFi.status() == WL_CONNECTED ? "WiFi OK" : "WiFi NOT CONNECTED");

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {

  // Subcribe for buzzer
  if (!client.connected()) reconnect();
  client.loop();

  long irValue = particleSensor.getIR();
  if (irValue > 7000 && checkForBeat(irValue)) {
    long delta = millis() - lastBeat;
    lastBeat = millis();

    float bpm = 60.0 / (delta / 1000.0);
    if (bpm < 255 && bpm > 20) {
      rates[rateSpot++] = (byte)bpm;
      rateSpot %= RATE_SIZE;
      hr = 0;
      for (byte i = 0; i < RATE_SIZE; i++) hr += rates[i];
      hr /= RATE_SIZE;
    }
  }

  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  sum_ax += ( a.acceleration.x / 9.8);
  sum_ay += ( a.acceleration.y / 9.8);
  sum_az += ( a.acceleration.z / 9.8);
  accCount++;

  // OLED
  if (millis() - lastOLEDMillis >= OLED_INTERVAL) {
    lastOLEDMillis = millis();
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(40, 28);
    display.printf("HR: %d", hr);

    if (isSleepSessionActive) {
      unsigned long s = (millis() - sleepStartMillis) / 1000;
      display.setTextSize(1);
      display.setCursor(20, 0);
      display.printf("%02lu:%02lu:%02lu", s/3600, (s/60)%60, s%60);
    }
    display.display();
  }

 //Publish
  if (isSleepSessionActive &&
      millis() - lastPublishMillis >= PUBLISH_INTERVAL &&
      accCount > 0) {

    lastPublishMillis = millis();

    String payload = "{";
    payload += "\"hr\":" + String(hr) + ",";
    payload += "\"ax\":" + String(sum_ax / accCount, 3) + ",";
    payload += "\"ay\":" + String(sum_ay / accCount, 3) + ",";
    payload += "\"az\":" + String(sum_az / accCount, 3) + "}";

    client.publish(TOPIC_SLEEP_DATA, payload.c_str(), false);

    sum_ax = sum_ay = sum_az = 0;
    accCount = 0;
  }

  static unsigned long pressStartMillis = 0;
  static bool handled = false;

  bool btn = digitalRead(BUTTON_PIN);

  if (btn == HIGH) {
    if (pressStartMillis == 0) {
      pressStartMillis = millis();
      handled = false;
    }

    if (!handled && millis() - pressStartMillis >= 19000) {
      handled = true;

      if (!isSleepSessionActive) {
        isSleepSessionActive = true;
        client.publish(TOPIC_SLEEP_BUTTON, "start_sleep");

        sleepStartMillis = millis();

        digitalWrite(BUZZER_PIN, HIGH);
        delay(150);
        digitalWrite(BUZZER_PIN, LOW);
      } else {
        isSleepSessionActive = false;
        client.publish(TOPIC_SLEEP_BUTTON, "wake_up");
      }
    }
  } 
  else {
    pressStartMillis = 0;
    handled = false;
  }
  delay(5);
}


