// ============================================================
// NeuroGuard v1
// Intelligent Wearable Safety Vest for Neurological Emergencies
// Team SenseSphere - IEEE YESIST12 2026
// FOCUS: Multi-condition detection + condition-aware physical intervention
// Target: ESP32 (BLE + I2C + SPI + PWM + FreeRTOS)
// ============================================================

#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP085.h>          // BMP180 - barometric pressure + temperature
#include <SparkFun_APDS9960.h>        // APDS9960 - proximity + gesture + ambient light
#include <MAX30105.h>
#include <heartRate.h>
#include <spo2_algorithm.h>

#include <SPI.h>
#include <SD.h>
#include <Preferences.h>

// =======================
// HARDWARE PINS
// =======================
#define SCREEN_WIDTH     128
#define SCREEN_HEIGHT    64
#define OLED_ADDR        0x3C
#define SD_CS            5
#define BTN_PIN          26     // Cancel push button (INPUT_PULLUP, active LOW)
#define BUZZER_PIN       25
#define PUMP1_PIN        32     // MOSFET gate - neck/head chamber
#define PUMP2_PIN        33     // MOSFET gate - torso chamber A
#define PUMP3_PIN        27     // MOSFET gate - torso chamber B
#define BATT_ADC_PIN     34
#define STATUS_LED       2

// I2C addresses on the shared bus
#define MPU6050_ADDR     0x69   // AD0 tied HIGH (alternate address, not the default 0x68)
#define MAX30102_ADDR    0x57
#define BMP180_ADDR      0x77
#define APDS9960_ADDR    0x39

// =======================
// OBJECTS
// =======================
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
Adafruit_MPU6050 mpu;
MAX30105         maxSensor;
Adafruit_BMP085  bmp;
SparkFun_APDS9960 apds = SparkFun_APDS9960();
Preferences      prefs;

BLEServer         *bleServer         = nullptr;
BLECharacteristic *telemetryChar     = nullptr;
BLECharacteristic *eventChar         = nullptr;
BLECharacteristic *commandChar       = nullptr;
BLECharacteristic *configChar        = nullptr;
bool               bleClientConnected = false;

// =======================
// CONFIG
// =======================
#define SAMPLE_PERIOD_MS       50      // 20 Hz sampling (matches NeuroGlove convention)
#define WINDOW_SIZE            20      // 1-second sliding window
#define CALIB_SAMPLES          100     // ~5 seconds @ 20Hz
#define LOG_FLUSH_EVERY        20      // flush SD every 1 s
#define ACK_TIMEOUT_MS         30000UL // 30 s cancel window
#define PUMP_INFLATE_MS        1500UL
#define PUMP_COOLDOWN_MS       10000UL
#define BUZZER_TONE_HZ         2800
#define AI_CONFIDENCE_THRESH   0.75f
#define FREE_FALL_G            0.4f
#define IMPACT_G               2.5f
#define ALTITUDE_DROP_M        0.4f    // barometric drop that corroborates a fall

// BLE UUIDs (custom service)
#define SVC_UUID          "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define TELEMETRY_UUID    "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
#define EVENT_UUID        "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
#define COMMAND_UUID      "6e400004-b5a3-f393-e0a9-e50e24dcca9e"
#define CONFIG_UUID       "6e400005-b5a3-f393-e0a9-e50e24dcca9e"

// Baseline thresholds (auto-tuned from calibration)
struct Thresholds {
  float stableMaxJerk;
  float tremorMinJerk;
  float freezingTremorMin;
  float freezingTremorMax;
  float seizureJerkMin;
  int   hrLowAbnormal;
  int   hrHighAbnormal;
  int   spo2LowAbnormal;
  float baselineAltitudeM;
  uint16_t proxSleepwalkWarn;
  float ambientDark;
} TH;

// =======================
// STATE
// =======================
enum NeuroState {
  ST_BOOT,
  ST_IDLE,
  ST_WALKING,
  ST_RESTING,
  ST_ACTIVITY,
  ST_SLEEPWALK,
  ST_SEIZURE,
  ST_FREEZING,
  ST_FALL,
  ST_SYNCOPE,
  ST_ACK_WINDOW,
  ST_INTERVENTION,
  ST_COOLDOWN
};

const char *stateName[] = {
  "BOOT", "IDLE", "WALKING", "RESTING", "ACTIVITY",
  "SLEEPWALK", "SEIZURE", "FREEZING", "FALL", "SYNCOPE",
  "CANCEL?", "INFLATE", "COOLDN"
};

NeuroState currentState  = ST_BOOT;
NeuroState detectedClass = ST_IDLE;
NeuroState lastAlerted   = ST_IDLE;

// Sensor buffers (WINDOW_SIZE samples @ 20 Hz = 1 s of history)
float axBuf[WINDOW_SIZE]       = {0};
float ayBuf[WINDOW_SIZE]       = {0};
float azBuf[WINDOW_SIZE]       = {0};
float gxBuf[WINDOW_SIZE]       = {0};
float gyBuf[WINDOW_SIZE]       = {0};
float gzBuf[WINDOW_SIZE]       = {0};
float accelMagBuf[WINDOW_SIZE] = {0};
float gyroMagBuf[WINDOW_SIZE]  = {0};
float jerkBuf[WINDOW_SIZE]     = {0};
float rollBuf[WINDOW_SIZE]     = {0};
float pitchBuf[WINDOW_SIZE]    = {0};
float hrBuf[WINDOW_SIZE]       = {0};
float spo2Buf[WINDOW_SIZE]     = {0};
float hrvBuf[WINDOW_SIZE]      = {0};
float pressBuf[WINDOW_SIZE]    = {0};   // BMP180 barometric pressure hPa
float tempBuf[WINDOW_SIZE]     = {0};   // BMP180 temperature C
float altBuf[WINDOW_SIZE]      = {0};   // derived altitude m
float proxBuf[WINDOW_SIZE]     = {0};   // APDS9960 proximity
float luxBuf[WINDOW_SIZE]      = {0};   // APDS9960 ambient light
int   bufIdx = 0;
bool  bufFilled = false;

// Calibration baselines
float baselineGyroMag  = 0;
float baselineAccelMag = 1.0f;
float baselineHR       = 70;
float baselineSpO2     = 97;
float baselinePressure = 1013.25f;
float baselineAltitude = 0.0f;
float baselineTemp     = 25.0f;
uint16_t baselineProx  = 0;
float baselineLux      = 200.0f;
bool  calibrated       = false;

// Complementary filter
float roll = 0, pitch = 0;
unsigned long lastFusionMs = 0;

// MAX30102 R-R interval detection
static const uint8_t RATE_SIZE = 8;
long   rrIntervals[RATE_SIZE] = {0};
uint8_t rrIdx = 0;
long    lastBeatMs = 0;
float   currentHR   = 0;
float   currentSpO2 = 0;
float   currentHRV  = 0;

// BMP180 latest values
float   currentPressure = 1013.25f;
float   currentTemp     = 25.0f;
float   currentAltitude = 0.0f;

// APDS9960 latest values
uint16_t currentProx    = 0;
uint16_t currentAmbient = 200;
uint16_t currentRed = 0, currentGreen = 0, currentBlue = 0;

// SpO2 rolling buffers
#define SPO2_BUF 100
uint32_t irBuf[SPO2_BUF];
uint32_t redBuf[SPO2_BUF];
uint16_t spo2Fill = 0;

// Sensor readiness
bool mpuReady = false, maxReady = false, sdReady = false, bleReady = false;
bool bmpReady = false, apdsReady = false;

// Runtime configuration (writable via BLE)
struct RuntimeConfig {
  float confThresh;
  uint32_t ackTimeoutMs;
  uint32_t inflateMs;
  bool enSleepwalk;
  bool enSeizure;
  bool enFreezing;
  bool enFall;
  bool enSyncope;
  uint8_t buzzerVolume;
} RTC = {AI_CONFIDENCE_THRESH, ACK_TIMEOUT_MS, PUMP_INFLATE_MS,
        true, true, true, true, true, 80};

// Session tracking
unsigned long lastSampleMs = 0;
unsigned long sessionStartMs = 0;
uint32_t sessionId = 0;
uint32_t sampleCount = 0;
uint8_t  batteryPct = 100;
volatile bool cancelRequested = false;
uint8_t currentPumpMask = 0;
float   lastConfidence = 0;

// SD batching
File   logFile;
String logBuffer;
int    logCounter = 0;

// Buzzer state
const int LEDC_CH = 0;
volatile bool buzzerActive = false;

// =======================
// HELPERS
// =======================
float meanf(float *a, int n) {
  float s = 0; for (int i = 0; i < n; i++) s += a[i]; return s / n;
}
float stdevf(float *a, int n, float m) {
  float s = 0; for (int i = 0; i < n; i++) { float d = a[i] - m; s += d * d; }
  return sqrtf(s / n);
}
float minf(float *a, int n) {
  float m = a[0]; for (int i = 1; i < n; i++) if (a[i] < m) m = a[i]; return m;
}
float maxf(float *a, int n) {
  float m = a[0]; for (int i = 1; i < n; i++) if (a[i] > m) m = a[i]; return m;
}
uint16_t zeroCrossings(float *a, int n, float ref) {
  uint16_t zc = 0; float last = a[0] - ref;
  for (int i = 1; i < n; i++) {
    float v = a[i] - ref;
    if ((v >= 0 && last < 0) || (v < 0 && last >= 0)) zc++;
    last = v;
  }
  return zc;
}

// =======================
// BLE JSON emitters
// =======================
void sendTelemetryJson(float gMag, float aMag, float jerk, NeuroState st) {
  if (!bleReady || !bleClientConnected || telemetryChar == nullptr) return;
  char buf[280];
  snprintf(buf, sizeof(buf),
    "{\"ts\":%lu,\"state\":\"%s\",\"cls\":%d,\"conf\":%.2f,"
    "\"hr\":%.1f,\"spo2\":%.1f,\"hrv\":%.1f,"
    "\"amag\":%.2f,\"gmag\":%.1f,\"jerk\":%.1f,"
    "\"roll\":%.1f,\"pitch\":%.1f,"
    "\"press\":%.1f,\"temp\":%.1f,\"alt\":%.2f,"
    "\"prox\":%u,\"lux\":%u,"
    "\"pumps\":%u,\"batt\":%u,\"sd\":%d}",
    (unsigned long)(millis() - sessionStartMs),
    stateName[st], (int)detectedClass, lastConfidence,
    currentHR, currentSpO2, currentHRV,
    aMag, gMag, jerk, roll, pitch,
    currentPressure, currentTemp, (currentAltitude - baselineAltitude),
    (unsigned)currentProx, (unsigned)currentAmbient,
    (unsigned)currentPumpMask, (unsigned)batteryPct, sdReady ? 1 : 0);
  telemetryChar->setValue((uint8_t*)buf, strlen(buf));
  telemetryChar->notify();
}

void sendEventJson(NeuroState cls, float conf, uint8_t mask, bool cancelled) {
  if (!bleReady || !bleClientConnected || eventChar == nullptr) return;
  char buf[220];
  snprintf(buf, sizeof(buf),
    "{\"ts\":%lu,\"cls\":%d,\"name\":\"%s\",\"conf\":%.2f,"
    "\"mask\":%u,\"canc\":%d,\"alt_drop\":%.2f,\"prox\":%u}",
    (unsigned long)(millis() - sessionStartMs),
    (int)cls, stateName[cls], conf,
    (unsigned)mask, cancelled ? 1 : 0,
    (currentAltitude - baselineAltitude), (unsigned)currentProx);
  eventChar->setValue((uint8_t*)buf, strlen(buf));
  eventChar->notify();
}

// =======================
// BLE CALLBACKS
// =======================
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *)    { bleClientConnected = true;  Serial.println("[BLE] client connected"); }
  void onDisconnect(BLEServer *) { bleClientConnected = false; BLEDevice::startAdvertising();
                                   Serial.println("[BLE] client disconnected -> re-advertising"); }
};

class CommandCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *ch) override {
    std::string v = ch->getValue();
    Serial.print("[BLE] cmd: "); Serial.println(v.c_str());
    if (v == "CANCEL")             cancelRequested = true;
    else if (v == "CALIBRATE")     { /* set flag - actual calib runs in loop */ }
    else if (v == "RESET_SESSION") { sessionId = (uint32_t)esp_random();
                                     sessionStartMs = millis(); sampleCount = 0; }
  }
};

class ConfigCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *ch) override {
    std::string v = ch->getValue();
    Serial.print("[BLE] cfg: "); Serial.println(v.c_str());
    char buf[200]; strncpy(buf, v.c_str(), sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
    char *tok = strtok(buf, ";");
    while (tok) {
      char key[32] = {0}; char val[32] = {0};
      if (sscanf(tok, "%31[^=]=%31s", key, val) == 2) {
        if      (!strcmp(key, "conf"))       RTC.confThresh   = atof(val);
        else if (!strcmp(key, "ack"))        RTC.ackTimeoutMs = strtoul(val, nullptr, 10);
        else if (!strcmp(key, "infl"))       RTC.inflateMs    = strtoul(val, nullptr, 10);
        else if (!strcmp(key, "en_sleep"))   RTC.enSleepwalk  = atoi(val) != 0;
        else if (!strcmp(key, "en_seiz"))    RTC.enSeizure    = atoi(val) != 0;
        else if (!strcmp(key, "en_freeze"))  RTC.enFreezing   = atoi(val) != 0;
        else if (!strcmp(key, "en_fall"))    RTC.enFall       = atoi(val) != 0;
        else if (!strcmp(key, "en_syncope")) RTC.enSyncope    = atoi(val) != 0;
        else if (!strcmp(key, "buzz_vol"))   RTC.buzzerVolume = (uint8_t)atoi(val);
      }
      tok = strtok(nullptr, ";");
    }
  }
};

// =======================
// BLE INITIALIZATION
// =======================
void initBLE() {
  char devName[32];
  uint64_t mac = ESP.getEfuseMac();
  snprintf(devName, sizeof(devName), "NeuroGuard-%04X", (uint16_t)(mac & 0xFFFF));

  BLEDevice::init(devName);
  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new ServerCallbacks());
  BLEService *svc = bleServer->createService(SVC_UUID);

  telemetryChar = svc->createCharacteristic(TELEMETRY_UUID,
                    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  telemetryChar->addDescriptor(new BLE2902());

  eventChar = svc->createCharacteristic(EVENT_UUID,
                BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  eventChar->addDescriptor(new BLE2902());

  commandChar = svc->createCharacteristic(COMMAND_UUID, BLECharacteristic::PROPERTY_WRITE);
  commandChar->setCallbacks(new CommandCallbacks());

  configChar = svc->createCharacteristic(CONFIG_UUID,
                 BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_READ);
  configChar->setCallbacks(new ConfigCallbacks());

  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SVC_UUID);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
  bleReady = true;
  Serial.printf("[BLE] Advertising as %s\n", devName);
}

// =======================
// BUZZER (LEDC PWM)
// =======================
void buzzerInit() {
  ledcSetup(LEDC_CH, BUZZER_TONE_HZ, 8);
  ledcAttachPin(BUZZER_PIN, LEDC_CH);
  ledcWrite(LEDC_CH, 0);
}
void buzzerOn()  { ledcWrite(LEDC_CH, map(RTC.buzzerVolume, 0, 100, 0, 255)); }
void buzzerOff() { ledcWrite(LEDC_CH, 0); }

void buzzerTask(void *pv) {
  bool on = false;
  while (true) {
    if (buzzerActive) {
      if (on) buzzerOff(); else buzzerOn();
      on = !on;
      vTaskDelay(pdMS_TO_TICKS(250));
    } else {
      buzzerOff();
      on = false;
      vTaskDelay(pdMS_TO_TICKS(100));
    }
  }
}

// =======================
// PUMP CONTROL
// =======================
void setPumps(uint8_t mask) {
  digitalWrite(PUMP1_PIN, (mask & 0b001) ? HIGH : LOW);
  digitalWrite(PUMP2_PIN, (mask & 0b010) ? HIGH : LOW);
  digitalWrite(PUMP3_PIN, (mask & 0b100) ? HIGH : LOW);
  currentPumpMask = mask;
}

uint8_t interventionMaskFor(NeuroState cls) {
  switch (cls) {
    case ST_SLEEPWALK: return 0b111;   // full upper body
    case ST_SEIZURE:   return 0b001;   // neck & head only
    case ST_FALL:      return 0b111;   // full upper body
    case ST_SYNCOPE:   return 0b111;   // full upper body
    case ST_FREEZING:  return 0b000;   // audio prompt only - no inflation
    default:           return 0b000;
  }
}

// =======================
// BMP180 + APDS9960 sampling
// =======================
void updateEnvironment() {
  if (bmpReady) {
    currentPressure = bmp.readPressure() / 100.0f;              // Pa -> hPa
    currentTemp     = bmp.readTemperature();
    currentAltitude = bmp.readAltitude(101325.0f);              // meters (uses sea-level 1013.25 hPa)
  }
  if (apdsReady) {
    uint8_t prox = 0;
    if (apds.readProximity(prox) == true) currentProx = prox;
    uint16_t ambient, red, green, blue;
    if (apds.readAmbientLight(ambient) &&
        apds.readRedLight(red) &&
        apds.readGreenLight(green) &&
        apds.readBlueLight(blue)) {
      currentAmbient = ambient;
      currentRed = red; currentGreen = green; currentBlue = blue;
    }
  }
}

// =======================
// CALIBRATION
// =======================
void runCalibration() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("NeuroGuard");
  display.println("Calibrating...");
  display.println("Wear vest still");
  display.display();

  float gSum = 0, aSum = 0, hrSum = 0, spo2Sum = 0;
  float pSum = 0, altSum = 0, tSum = 0;
  uint32_t proxSum = 0; uint32_t luxSum = 0;
  float gSamples[CALIB_SAMPLES];

  for (int i = 0; i < CALIB_SAMPLES; i++) {
    sensors_event_t a, g, t;
    if (mpuReady) mpu.getEvent(&a, &g, &t);
    float gMag = fabsf(g.gyro.x) + fabsf(g.gyro.y) + fabsf(g.gyro.z);
    float aMag = sqrtf(a.acceleration.x*a.acceleration.x +
                       a.acceleration.y*a.acceleration.y +
                       a.acceleration.z*a.acceleration.z) / 9.80665f;
    gSamples[i] = gMag;
    gSum += gMag; aSum += aMag;
    hrSum += currentHR; spo2Sum += currentSpO2;

    updateEnvironment();
    pSum   += currentPressure;
    altSum += currentAltitude;
    tSum   += currentTemp;
    proxSum += currentProx;
    luxSum  += currentAmbient;

    display.fillRect(0, 48, 128, 16, BLACK);
    display.setCursor(0, 48);
    display.print("Sample "); display.print(i+1); display.print("/");
    display.println(CALIB_SAMPLES);
    display.display();
    delay(50);
  }

  baselineGyroMag  = gSum / CALIB_SAMPLES;
  baselineAccelMag = aSum / CALIB_SAMPLES;
  baselineHR       = (hrSum / CALIB_SAMPLES > 30) ? hrSum / CALIB_SAMPLES : 70;
  baselineSpO2     = (spo2Sum / CALIB_SAMPLES > 80) ? spo2Sum / CALIB_SAMPLES : 97;
  baselinePressure = pSum   / CALIB_SAMPLES;
  baselineAltitude = altSum / CALIB_SAMPLES;
  baselineTemp     = tSum   / CALIB_SAMPLES;
  baselineProx     = proxSum / CALIB_SAMPLES;
  baselineLux      = luxSum  / CALIB_SAMPLES;
  float gSD = stdevf(gSamples, CALIB_SAMPLES, baselineGyroMag);

  // Adaptive thresholds derived from wearer's own baseline
  TH.stableMaxJerk     = baselineGyroMag + 3 * gSD;
  TH.tremorMinJerk     = fmaxf(0.05f, 8 * gSD);
  TH.freezingTremorMin = 4.0f;
  TH.freezingTremorMax = 8.0f;
  TH.seizureJerkMin    = fmaxf(2.5f, 30 * gSD);
  TH.hrLowAbnormal     = fmaxf(40.0f, baselineHR - 25);
  TH.hrHighAbnormal    = fminf(180.0f, baselineHR + 60);
  TH.spo2LowAbnormal   = 88;
  TH.baselineAltitudeM = baselineAltitude;
  TH.proxSleepwalkWarn = 60;                                   // APDS9960 proximity units
  TH.ambientDark       = fmaxf(20.0f, baselineLux * 0.15f);    // "dark room" threshold

  prefs.begin("neuroguard", false);
  prefs.putFloat("bGyro",    baselineGyroMag);
  prefs.putFloat("bAccel",   baselineAccelMag);
  prefs.putFloat("bHR",      baselineHR);
  prefs.putFloat("bSpO2",    baselineSpO2);
  prefs.putFloat("bPress",   baselinePressure);
  prefs.putFloat("bAlt",     baselineAltitude);
  prefs.putFloat("bTemp",    baselineTemp);
  prefs.putUShort("bProx",   baselineProx);
  prefs.putFloat("bLux",     baselineLux);
  prefs.putFloat("thStable", TH.stableMaxJerk);
  prefs.putFloat("thTremor", TH.tremorMinJerk);
  prefs.putFloat("thSeiz",   TH.seizureJerkMin);
  prefs.putInt  ("thHrLo",   TH.hrLowAbnormal);
  prefs.putInt  ("thHrHi",   TH.hrHighAbnormal);
  prefs.end();

  calibrated = true;
  Serial.printf("[CAL] gBase=%.3f aBase=%.3f HR=%.0f SpO2=%.0f\n",
                baselineGyroMag, baselineAccelMag, baselineHR, baselineSpO2);
  Serial.printf("[CAL] Press=%.1fhPa Alt=%.1fm Temp=%.1fC Prox=%u Lux=%.0f\n",
                baselinePressure, baselineAltitude, baselineTemp,
                (unsigned)baselineProx, baselineLux);
}

void loadCalibration() {
  prefs.begin("neuroguard", true);
  baselineGyroMag  = prefs.getFloat("bGyro", 0);
  baselineAccelMag = prefs.getFloat("bAccel", 1.0f);
  baselineHR       = prefs.getFloat("bHR", 70);
  baselineSpO2     = prefs.getFloat("bSpO2", 97);
  baselinePressure = prefs.getFloat("bPress", 1013.25f);
  baselineAltitude = prefs.getFloat("bAlt", 0.0f);
  baselineTemp     = prefs.getFloat("bTemp", 25.0f);
  baselineProx     = prefs.getUShort("bProx", 0);
  baselineLux      = prefs.getFloat("bLux", 200.0f);
  TH.stableMaxJerk     = prefs.getFloat("thStable", 1.0f);
  TH.tremorMinJerk     = prefs.getFloat("thTremor", 2.5f);
  TH.seizureJerkMin    = prefs.getFloat("thSeiz",   2.5f);
  TH.freezingTremorMin = 4.0f;
  TH.freezingTremorMax = 8.0f;
  TH.hrLowAbnormal     = prefs.getInt("thHrLo", 45);
  TH.hrHighAbnormal    = prefs.getInt("thHrHi", 160);
  TH.spo2LowAbnormal   = 88;
  TH.baselineAltitudeM = baselineAltitude;
  TH.proxSleepwalkWarn = 60;
  TH.ambientDark       = fmaxf(20.0f, baselineLux * 0.15f);
  calibrated = prefs.getFloat("bGyro", -1) > 0;
  prefs.end();
}

// =======================
// SD LOGGING
// =======================
void initLog() {
  if (!sdReady) return;
  bool isNew = !SD.exists("/neuroguard_log.csv");
  logFile = SD.open("/neuroguard_log.csv", FILE_APPEND);
  if (logFile && isNew) {
    logFile.println("session,t_ms,ax,ay,az,gx,gy,gz,gyroMag,accelMag,jerk,"
                    "roll,pitch,hr,spo2,hrv,"
                    "press,temp,alt,prox,lux,red,green,blue,"
                    "cls,conf,mask,cancel,batt,state");
    logFile.close();
  } else if (logFile) {
    logFile.close();
  }
}

void appendLog(unsigned long t, float ax, float ay, float az,
               float gx, float gy, float gz,
               float gMag, float aMag, float jerk,
               NeuroState cls, float conf, uint8_t mask,
               bool cancelled, NeuroState st) {
  if (!sdReady) return;
  logBuffer += String(sessionId) + ",";
  logBuffer += String(t)         + ",";
  logBuffer += String(ax, 3)     + "," + String(ay, 3) + "," + String(az, 3) + ",";
  logBuffer += String(gx, 2)     + "," + String(gy, 2) + "," + String(gz, 2) + ",";
  logBuffer += String(gMag, 2)   + "," + String(aMag, 3) + "," + String(jerk, 2) + ",";
  logBuffer += String(roll, 1)   + "," + String(pitch, 1) + ",";
  logBuffer += String(currentHR, 1)  + "," + String(currentSpO2, 1) + "," + String(currentHRV, 1) + ",";
  logBuffer += String(currentPressure, 2) + "," + String(currentTemp, 2) + ",";
  logBuffer += String(currentAltitude - baselineAltitude, 3) + ",";
  logBuffer += String(currentProx)    + "," + String(currentAmbient) + ",";
  logBuffer += String(currentRed)     + "," + String(currentGreen)   + "," + String(currentBlue) + ",";
  logBuffer += String((int)cls)  + "," + String(conf, 3) + ",";
  logBuffer += String(mask)      + "," + String(cancelled ? 1 : 0) + ",";
  logBuffer += String(batteryPct) + "," + String(stateName[st]) + "\n";
  logCounter++;
  if (logCounter >= LOG_FLUSH_EVERY) {
    logFile = SD.open("/neuroguard_log.csv", FILE_APPEND);
    if (logFile) { logFile.print(logBuffer); logFile.close(); }
    logBuffer = "";
    logCounter = 0;
  }
}

// =======================
// NEUROLOGICAL CLASSIFIER
// Rule-based safety baseline (runs ALWAYS; ML overlays when confident)
// =======================
NeuroState classify(float aMag, float gMag, float jerk,
                    float hr, float spo2, float hrv,
                    float amagMin, float amagMax, float amagMean,
                    uint16_t zcAccel, float hrSlope, float stillFrac,
                    float freeFallFrac, float altDrop,
                    uint16_t prox, uint16_t ambient, float *outConf) {

  // ---- Free-fall + impact + stillness + barometric drop -> FALL ----
  //  BMP180 altitude drop is used as an INDEPENDENT physical corroboration
  //  of a genuine fall: a real fall drops the wearer by >~0.4 m.
  if (freeFallFrac > 0.15f && amagMax > IMPACT_G && stillFrac > 0.3f) {
    float conf = 0.90f;
    if (altDrop > ALTITUDE_DROP_M) conf = 0.97f;   // strong corroboration
    *outConf = conf;
    return ST_FALL;
  }

  // ---- Rapid HR drop + gradual accel decline -> SYNCOPE ----
  if (hrSlope < -15.0f && spo2 > 0 && spo2 < 95 && amagMean < 0.9f) {
    *outConf = 0.88f;
    return ST_SYNCOPE;
  }

  // ---- High-frequency high-amplitude oscillation -> SEIZURE ----
  if (jerk > TH.seizureJerkMin && zcAccel > 25 &&
      hr > TH.hrHighAbnormal - 20) {
    *outConf = 0.92f;
    return ST_SEIZURE;
  }

  // ---- Low-amplitude 4-8 Hz tremor with stationary base -> FREEZING ----
  if (zcAccel >= (uint16_t)(2 * TH.freezingTremorMin) &&
      zcAccel <= (uint16_t)(2 * TH.freezingTremorMax) &&
      (amagMax - amagMin) < 0.3f && stillFrac > 0.4f) {
    *outConf = 0.85f;
    return ST_FREEZING;
  }

  // ---- Slow steady gait at low HR + DARK ambient (sleep) -> SLEEPWALKING ----
  //  APDS9960 ambient light acts as an INDEPENDENT nocturnal gate:
  //  we only trigger sleepwalking in a dark room to reject false daytime hits.
  //  Proximity buildup adds urgency (collision imminent).
  bool nocturnal = ((float)ambient < TH.ambientDark);
  bool proxWarn  = (prox > TH.proxSleepwalkWarn);
  if (gMag > baselineGyroMag * 1.5f && gMag < baselineGyroMag * 3.5f &&
      hr > 0 && hr < baselineHR * 0.9f && hrv > 60.0f && nocturnal) {
    *outConf = proxWarn ? 0.90f : 0.80f;
    return ST_SLEEPWALK;
  }

  // ---- Normal-ish activity classification ----
  if (jerk > TH.stableMaxJerk * 2)     { *outConf = 0.70f; return ST_ACTIVITY; }
  if (gMag > baselineGyroMag * 1.8f)   { *outConf = 0.75f; return ST_WALKING; }
  if (stillFrac > 0.7f)                { *outConf = 0.85f; return ST_RESTING; }
  *outConf = 0.60f;
  return ST_IDLE;
}

bool isEmergency(NeuroState s) {
  return s == ST_SLEEPWALK || s == ST_SEIZURE || s == ST_FREEZING ||
         s == ST_FALL || s == ST_SYNCOPE;
}

bool classEnabled(NeuroState s) {
  switch (s) {
    case ST_SLEEPWALK: return RTC.enSleepwalk;
    case ST_SEIZURE:   return RTC.enSeizure;
    case ST_FREEZING:  return RTC.enFreezing;
    case ST_FALL:      return RTC.enFall;
    case ST_SYNCOPE:   return RTC.enSyncope;
    default:           return true;
  }
}

// =======================
// INTERVENTION FLOW
// =======================
void runInterventionFlow(NeuroState cls, float conf) {
  cancelRequested = false;
  currentState = ST_ACK_WINDOW;
  buzzerActive = true;

  if (cls == ST_FREEZING) {
    Serial.println("[INT] Freezing detected - audio prompt only");
    uint32_t start = millis();
    while ((millis() - start) < RTC.ackTimeoutMs) {
      if (cancelRequested || digitalRead(BTN_PIN) == LOW) break;
      delay(50);
    }
    buzzerActive = false;
    sendEventJson(cls, conf, 0, cancelRequested);
    currentState = ST_COOLDOWN;
    delay(PUMP_COOLDOWN_MS);
    currentState = ST_IDLE;
    return;
  }

  bool preemptive = (cls == ST_SYNCOPE || cls == ST_FALL);
  uint32_t effectiveAck = preemptive ? 3000UL : RTC.ackTimeoutMs;

  uint32_t start = millis();
  while ((millis() - start) < effectiveAck) {
    if (cancelRequested || digitalRead(BTN_PIN) == LOW) {
      cancelRequested = true;
      break;
    }
    uint32_t remaining = effectiveAck - (millis() - start);
    display.fillRect(0, 54, 128, 10, BLACK);
    display.setCursor(0, 54);
    display.print("CANCEL in ");
    display.print(remaining / 1000);
    display.print("s");
    display.display();
    delay(100);
  }

  uint8_t mask = 0;
  bool cancelled = cancelRequested;

  if (!cancelled) {
    mask = interventionMaskFor(cls);
    currentState = ST_INTERVENTION;
    Serial.printf("[INT] Inflating mask=0b%03u for class=%s\n",
                  (unsigned)mask, stateName[cls]);
    setPumps(mask);
    delay(RTC.inflateMs);
    setPumps(0);
  } else {
    Serial.println("[INT] User cancelled - no inflation");
  }
  buzzerActive = false;

  sendEventJson(cls, conf, mask, cancelled);

  currentState = ST_COOLDOWN;
  delay(PUMP_COOLDOWN_MS);
  currentState = ST_IDLE;
}

// =======================
// MAX30102 PROCESSING
// =======================
void updatePhysio() {
  if (!maxReady) return;
  maxSensor.check();
  while (maxSensor.available()) {
    uint32_t ir  = maxSensor.getIR();
    uint32_t red = maxSensor.getRed();
    maxSensor.nextSample();

    for (int i = 1; i < SPO2_BUF; i++) {
      irBuf[i - 1]  = irBuf[i];
      redBuf[i - 1] = redBuf[i];
    }
    irBuf[SPO2_BUF - 1]  = ir;
    redBuf[SPO2_BUF - 1] = red;
    if (spo2Fill < SPO2_BUF) spo2Fill++;

    if (checkForBeat(ir)) {
      long now = millis();
      long delta = now - lastBeatMs;
      lastBeatMs = now;
      if (delta > 300 && delta < 2000) {
        rrIntervals[rrIdx++] = delta;
        rrIdx %= RATE_SIZE;
        currentHR = 60000.0f / delta;
      }
    }

    float sumSq = 0; int n = 0;
    for (int i = 1; i < RATE_SIZE; i++) {
      if (rrIntervals[i] > 0 && rrIntervals[i - 1] > 0) {
        float d = (float)(rrIntervals[i] - rrIntervals[i - 1]);
        sumSq += d * d; n++;
      }
    }
    currentHRV = n ? sqrtf(sumSq / n) : 0;

    if (spo2Fill >= SPO2_BUF) {
      int32_t spo2Val; int8_t validSpO2;
      int32_t hrVal;   int8_t validHR;
      maxim_heart_rate_and_oxygen_saturation(
        irBuf, SPO2_BUF, redBuf, &spo2Val, &validSpO2, &hrVal, &validHR);
      if (validSpO2) currentSpO2 = (float)spo2Val;
    }

    if (ir < 50000) {
      currentHR = 0; currentSpO2 = 0; currentHRV = 0;
    }
  }
}

// =======================
// BATTERY MONITOR
// =======================
void updateBattery() {
  uint32_t sum = 0;
  for (int i = 0; i < 8; i++) { sum += analogRead(BATT_ADC_PIN); delayMicroseconds(200); }
  float vmv = (sum / 8.0f) / 4095.0f * 3300.0f * 2.0f;
  if      (vmv <= 3300) batteryPct = 0;
  else if (vmv >= 4200) batteryPct = 100;
  else batteryPct = (uint8_t)((vmv - 3300) * 100.0f / 900.0f);
}

// =======================
// SETUP
// =======================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n============================================================");
  Serial.println(" NeuroGuard v1 - Team SenseSphere - YESIST12 2026");
  Serial.println("============================================================");

  pinMode(BTN_PIN,      INPUT_PULLUP);
  pinMode(PUMP1_PIN,    OUTPUT);
  pinMode(PUMP2_PIN,    OUTPUT);
  pinMode(PUMP3_PIN,    OUTPUT);
  pinMode(STATUS_LED,   OUTPUT);
  pinMode(BATT_ADC_PIN, INPUT);
  analogReadResolution(12);
  setPumps(0);
  buzzerInit();

  Wire.begin(21, 22, 400000UL);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED FAILED"); while (true) delay(1000);
  }
  display.clearDisplay();
  display.setTextColor(WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("NeuroGuard v1");
  display.println("Booting...");
  display.display();

  // MPU6050 at alternate I2C address 0x69 (AD0 tied HIGH)
  mpuReady = mpu.begin(MPU6050_ADDR);
  if (mpuReady) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
    Serial.println("MPU6050 OK @ 0x69");
  } else Serial.println("MPU6050 FAIL @ 0x69");

  if (maxSensor.begin(Wire, I2C_SPEED_FAST)) {
    maxSensor.setup(0x1F, 4, 2, 100, 411, 4096);
    maxSensor.setPulseAmplitudeRed(0x0A);
    maxSensor.setPulseAmplitudeIR(0x0A);
    maxReady = true;
    Serial.println("MAX30102 OK");
  } else Serial.println("MAX30102 FAIL");

  // BMP180 barometric pressure sensor
  bmpReady = bmp.begin();
  Serial.println(bmpReady ? "BMP180 OK @ 0x77" : "BMP180 FAIL");

  // APDS9960 proximity + gesture + ambient light sensor
  apdsReady = apds.init();
  if (apdsReady) {
    apds.enableLightSensor(false);      // ambient light + RGB
    apds.enableProximitySensor(false);  // proximity
    apds.setProximityGain(PGAIN_2X);
    Serial.println("APDS9960 OK @ 0x39");
  } else Serial.println("APDS9960 FAIL");

  sdReady = SD.begin(SD_CS);
  Serial.println(sdReady ? "SD OK" : "SD FAIL");
  if (sdReady) initLog();

  initBLE();
  loadCalibration();

  if (digitalRead(BTN_PIN) == LOW || !calibrated) runCalibration();

  xTaskCreatePinnedToCore(buzzerTask, "buzz", 2048, nullptr, 3, nullptr, 1);

  sessionId = (uint32_t)esp_random();
  sessionStartMs = millis();
  lastSampleMs = millis();
  lastFusionMs = millis();
  updateBattery();

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("NeuroGuard READY");
  display.print("Session "); display.println(sessionId);
  display.display();

  currentState = ST_IDLE;
  Serial.printf("[SYS] Session %u started\n", sessionId);
}

// =======================
// LOOP
// =======================
void loop() {
  updatePhysio();
  updateEnvironment();

  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_PERIOD_MS) return;
  float dt = (now - lastSampleMs) / 1000.0f;
  lastSampleMs = now;

  // ---- READ MPU ----
  sensors_event_t a, g, t;
  if (mpuReady) mpu.getEvent(&a, &g, &t);

  float ax = a.acceleration.x / 9.80665f;
  float ay = a.acceleration.y / 9.80665f;
  float az = a.acceleration.z / 9.80665f;
  float gx = g.gyro.x * 57.2958f;
  float gy = g.gyro.y * 57.2958f;
  float gz = g.gyro.z * 57.2958f;

  float gyroMag  = fabsf(gx) + fabsf(gy) + fabsf(gz);
  float accelMag = sqrtf(ax*ax + ay*ay + az*az);

  float accRoll  = atan2f(ay, az) * 57.2958f;
  float accPitch = atan2f(-ax, sqrtf(ay*ay + az*az)) * 57.2958f;
  roll  = 0.96f * (roll  + gx * dt) + 0.04f * accRoll;
  pitch = 0.96f * (pitch + gy * dt) + 0.04f * accPitch;

  float lastMag = bufIdx == 0 ? accelMagBuf[WINDOW_SIZE - 1] : accelMagBuf[bufIdx - 1];
  float jerk = fabsf(accelMag - lastMag) / fmaxf(dt, 0.001f);

  // ---- BUFFER ----
  axBuf[bufIdx] = ax; ayBuf[bufIdx] = ay; azBuf[bufIdx] = az;
  gxBuf[bufIdx] = gx; gyBuf[bufIdx] = gy; gzBuf[bufIdx] = gz;
  accelMagBuf[bufIdx] = accelMag;
  gyroMagBuf[bufIdx]  = gyroMag;
  jerkBuf[bufIdx]     = jerk;
  rollBuf[bufIdx]     = roll;
  pitchBuf[bufIdx]    = pitch;
  hrBuf[bufIdx]       = currentHR;
  spo2Buf[bufIdx]     = currentSpO2;
  hrvBuf[bufIdx]      = currentHRV;
  pressBuf[bufIdx]    = currentPressure;
  tempBuf[bufIdx]     = currentTemp;
  altBuf[bufIdx]      = currentAltitude;
  proxBuf[bufIdx]     = currentProx;
  luxBuf[bufIdx]      = currentAmbient;

  bufIdx = (bufIdx + 1) % WINDOW_SIZE;
  if (bufIdx == 0) bufFilled = true;

  int n = bufFilled ? WINDOW_SIZE : bufIdx;

  // ---- WINDOW FEATURES ----
  float aMean   = meanf(accelMagBuf, n);
  float gMean   = meanf(gyroMagBuf,  n);
  float jMean   = meanf(jerkBuf,     n);
  float aMin    = minf(accelMagBuf,  n);
  float aMax    = maxf(accelMagBuf,  n);
  uint16_t zcA  = zeroCrossings(accelMagBuf, n, 1.0f);

  int stillCount = 0, freeFallCount = 0;
  for (int i = 0; i < n; i++) {
    if (fabsf(accelMagBuf[i] - 1.0f) < 0.1f) stillCount++;
    if (accelMagBuf[i] < FREE_FALL_G)        freeFallCount++;
  }
  float stillFrac    = (float)stillCount    / n;
  float freeFallFrac = (float)freeFallCount / n;

  float hrSlope = 0;
  if (n >= 5) hrSlope = (hrBuf[bufIdx == 0 ? n - 1 : bufIdx - 1] -
                         hrBuf[bufIdx]) * 20.0f / n;

  // Altitude drop within the current window (baseline-referenced)
  float altDrop = maxf(altBuf, n) - minf(altBuf, n);

  // ---- CLASSIFY ----
  float conf = 0;
  NeuroState cls = classify(aMean, gMean, jMean,
                            currentHR, currentSpO2, currentHRV,
                            aMin, aMax, aMean, zcA, hrSlope,
                            stillFrac, freeFallFrac, altDrop,
                            currentProx, currentAmbient, &conf);
  detectedClass  = cls;
  lastConfidence = conf;

  // ---- OLED ----
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("NG "); display.print(stateName[currentState]);
  display.setCursor(96, 0);
  display.print(batteryPct); display.print("%");

  display.setTextSize(2);
  display.setCursor(0, 12);
  display.print(stateName[cls]);

  display.setTextSize(1);
  display.setCursor(0, 34);
  display.print("conf "); display.print(conf, 2);
  display.setCursor(0, 44);
  display.print("HR "); display.print((int)currentHR);
  display.print(" SpO2 "); display.print((int)currentSpO2);

  if (currentState == ST_IDLE) {
    display.setCursor(0, 54);
    display.print("P"); display.print((int)currentPressure);
    display.print(" Px"); display.print(currentProx);
    display.print(" L"); display.print(currentAmbient);
  }
  display.display();

  // ---- TELEMETRY + LOG ----
  sendTelemetryJson(gMean, aMean, jMean, currentState);
  appendLog(now - sessionStartMs, ax, ay, az, gx, gy, gz,
            gMean, aMean, jMean, cls, conf, currentPumpMask,
            false, currentState);

  sampleCount++;

  // ---- EMERGENCY DISPATCH ----
  if (bufFilled && isEmergency(cls) && classEnabled(cls) &&
      conf >= RTC.confThresh && currentState == ST_IDLE) {
    runInterventionFlow(cls, conf);
  }

  if (sampleCount % 100 == 0) updateBattery();

  digitalWrite(STATUS_LED, (millis() / 500) % 2);
}
