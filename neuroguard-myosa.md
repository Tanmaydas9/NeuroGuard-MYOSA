publishDate: 2026-08-01T00:00:00Z

title: NeuroGuard - Intelligent Wearable Safety Vest for Neurological Emergencies

excerpt: AI-assisted wearable vest for real-time detection and physical intervention during neurological emergencies including seizures, sleepwalking, freezing of gait, syncope, and sudden falls.

image: neuroguard-cover.jpg

tags:
  - healthcare
  - ai
  - embedded-systems
  - wearable-tech
  - neurological-safety
  - emergency-response

> An AI-driven wearable safety vest that predicts, detects, and physically intervenes during neurological emergencies before injury occurs.

## Overview
NeuroGuard is an intelligent wearable safety vest developed by Team SenseSphere as an evolutionary upgrade of our previous MYOSA rehabilitation platform. While NeuroGlove focused on motor coordination monitoring, NeuroGuard extends the same embedded-AI + multimodal-sensing philosophy to a new domain: real-time detection and physical intervention during acute neurological emergencies.

The system combines:
- inertial motion sensing,
- physiological signal analysis,
- environmental sensing,
- proximity and ambient light awareness,
- embedded artificial intelligence,
- temporal behavioral inference,
- multi-condition classification,
- and rapid mechanical intervention

to protect individuals living with neurological disorders such as:
- Somnambulism (Sleepwalking),
- Epileptic Seizures,
- Parkinson's Freezing of Gait,
- Sudden Falls,
- and Syncope (Fainting).

NeuroGuard continuously studies:
- three-axis acceleration and angular velocity,
- gait characteristics and tremor oscillations,
- heart rate and heart rate variability,
- blood oxygen saturation,
- posture and orientation dynamics,
- barometric pressure and altitude changes,
- proximity to obstacles and ambient light context,
- and temporal physiological consistency

to provide condition-aware protective intervention during acute neurological events.

The wearable integrates:
- ESP32 microcontroller
- MPU6050 inertial measurement unit (I2C addr 0x69)
- MAX30102 pulse oximeter and heart rate sensor
- BMP180 barometric pressure and temperature sensor
- APDS9960 proximity, gesture and ambient light sensor
- OLED visualization display
- SD-card event logging
- Active buzzer emergency alert
- Cancel push button acknowledgement
- Three MOSFET-driven 24V air pumps
- PVC inflatable protective chambers
- Bluetooth Low Energy mobile interface
- and deep-learning-based neurological state inference.

## Features

- Real-time neurological emergency detection
- Five-condition classification (Sleepwalking, Seizure, Freezing of Gait, Fall, Syncope)
- On-device TensorFlow Lite inference
- Condition-aware physical intervention
- 30-second user acknowledgement window
- OLED-based live visualization
- SD-card behavioral event logging
- Bluetooth Low Energy telemetry
- MOSFET-driven pump actuation
- Adaptive baseline calibration
- Confidence-gated prediction rejection
- Multimodal sensor fusion (motion + physiological + environmental)
- Barometric altitude change detection (fall corroboration)
- Proximity-aware sleepwalking collision warning
- Ambient light context (sleep vs waking detection)
- Temporal deep-learning behavioral analysis
- Automatic buzzer alarm with volume control
- Battery monitoring and reporting
- Modular architecture for future conditions

## Tech Stack

### Hardware
- ESP32
- MPU6050 (I2C 0x69)
- MAX30102 (I2C 0x57)
- BMP180 (I2C 0x77)
- APDS9960 (I2C 0x39)
- OLED Display SSD1306 (I2C 0x3C)
- MicroSD Card Module (SPI)
- Active Buzzer
- Cancel Push Button
- 3 x IRLZ44N MOSFET drivers
- 3 x 24V miniature air pumps
- PVC inflatable chambers
- 24V DC adapter (pump rail)
- 3.7V Li-ion cell + TP4056 (logic rail)

### Software
- Arduino IDE
- Embedded C/C++
- FreeRTOS multitasking
- MIT App Inventor / Flutter
- Bluetooth Low Energy Communication

### AI & Analysis
- Signal Processing
- Sensor Fusion
- Classical Machine Learning
- Temporal Deep Learning
- CNN + BiLSTM + Attention
- TensorFlow Lite for Microcontrollers
- Confidence-based Behavioral Inference

## Working Principle

After initialization, NeuroGuard begins an initial calibration phase where baseline movement, physiological, and environmental characteristics are recorded for the specific wearer.

The system continuously collects synchronized sensor data including:
- three-axis acceleration and gyroscope,
- accel magnitude and jerk,
- roll and pitch orientation,
- heart rate and HRV,
- blood oxygen saturation,
- barometric pressure and derived altitude,
- ambient temperature,
- proximity to nearby objects,
- ambient light (lux) and RGB color context,
- and pulse amplitude.

The behavioral-analysis pipeline studies:
- gait continuity,
- tremor frequency signatures,
- impact and free-fall dynamics,
- barometric altitude drop (fall verification),
- physiological perturbations,
- proximity buildup (imminent collision during sleepwalking),
- ambient light context (nocturnal vs diurnal disambiguation),
- and temporal event consistency.

Instead of reacting to isolated sensor readings, the system analyzes short temporal behavioral windows (one full second at 20 Hz) to differentiate:
- normal walking,
- resting states,
- daily activity,
- somnambulism episodes,
- epileptic seizures,
- freezing of gait,
- sudden falls,
- and syncope collapses.

When a neurological emergency is detected with sufficient confidence, the system enters an acknowledgement state. The OLED displays the diagnosis, the buzzer sounds, and a 30-second countdown begins. If the wearer is conscious, pressing the cancel button aborts intervention. Otherwise, the correct pump mask fires for the detected condition:

- Sleepwalking : upper-body inflation to cushion collisions
- Seizure : neck and head chamber inflation
- Fall : full upper-body inflation on free-fall + impact + barometric drop signature
- Syncope : preemptive inflation on the physiological warning
- Freezing of Gait : audio prompts only, unless a subsequent fall is detected

Based on the analyzed neurological state, the wearable also generates real-time telemetry through the connected mobile interface for caregiver awareness.

## Advancements Over Previous MYOSA Version (NeuroGlove)

NeuroGuard is an upgraded evolution of the NeuroGlove MYOSA project. The prior version focused on rehabilitation coaching for motor coordination disorders. NeuroGuard reuses the same embedded-AI backbone but redirects it toward emergency safety with the following new features and technical improvements:

### New Features
- Multi-condition neurological classification (5 emergency + 3 normal classes) replacing single-domain rehab state machine.
- Dual-modality sensor fusion with the MAX30102 pulse oximeter added alongside the MPU6050 for physiological signatures.
- Environmental context sensing with BMP180 for barometric altitude verification of falls.
- Proximity and ambient light awareness through APDS9960 for sleepwalking collision warnings and nocturnal disambiguation.
- Physical intervention subsystem with three MOSFET-driven 24V air pumps and PVC inflatable chambers - the previous version was alert-only.
- 30-second user acknowledgement window with hardware cancel button for false-alarm suppression.
- Condition-specific inflation policy - freezing of gait deliberately delays inflation, syncope inflates preemptively, seizure targets neck/head only.
- On-device TensorFlow Lite for Microcontrollers inference for closed-loop emergency response with no phone dependency.

### Technical Improvements
- FreeRTOS multitasking replacing the single-loop v2 firmware; 10 concurrent tasks pinned across both ESP32 cores.
- BLE GATT service with JSON payloads replacing classic BluetoothSerial SPP, enabling cross-platform Flutter clients.
- Confidence-gated inference - predictions below a runtime-adjustable threshold are rejected, dramatically reducing false interventions.
- Structured daily-rotating SD CSV logs (event log + intervention outcome) versus flat session logs.
- End-to-end retraining pipeline: synthetic dataset generation for 8 classes with 16 input features, CNN+BiLSTM+Attention training, FP16 and INT8 TFLite export, C-header embedding.
- Runtime-tunable configuration (confidence threshold, ack timeout, per-condition enable/disable, buzzer volume) writeable over BLE.
- MPU6050 wired at alternate address 0x69 (AD0 tied HIGH) to avoid future I2C address collisions on the shared bus.

## Future Scope

Future development objectives for NeuroGuard include:

- Compressed CO2 cartridge inflation for automotive-airbag-speed deployment (~30ms)
- TPU inflatable bladders replacing PVC chambers for daily-wear form factor
- Additional neurological conditions (myoclonic jerks, dystonia, cataplexy)
- Predictive pre-ictal seizure detection using EEG add-on
- Cloud-based longitudinal patient dashboard
- Multi-caregiver Firebase Cloud Messaging emergency notifications
- Personalized on-device model retraining from SD-card logs
- Integration with emergency medical services API
- Clinical validation with neurology departments
- Insurance and healthcare compliance certification
- Pediatric and geriatric variants
- Waterproof and shock-proof enclosure

## SDG Alignment

NeuroGuard aligns with the following United Nations Sustainable Development Goals (SDGs):

- SDG 3: Good Health and Well-Being
- SDG 9: Industry, Innovation and Infrastructure
- SDG 10: Reduced Inequalities

The project aims to contribute toward accessible, intelligent, and life-saving assistive technology for individuals living with neurological disorders.

## Installation & Usage

1. Assemble the hardware components and connect sensors to the ESP32 microcontroller following the wiring in `neuroguard_main.ino`. Note that MPU6050's AD0 pin is tied HIGH so its I2C address is `0x69`, not the default `0x68`.
2. Install required Arduino libraries: Adafruit_MPU6050, Adafruit_SSD1306, SparkFun_MAX3010x, Adafruit_BMP085 (BMP180), SparkFun_APDS9960, ArduinoBLE, ArduinoJson, TensorFlowLite_ESP32.
3. Run `python generate_synthetic_dataset.py` to bootstrap the training data.
4. Run `python neuroguard_model_training.py` to train the CNN+BiLSTM+Attention model and export the TFLite artifacts.
5. Convert the exported `.tflite` to a C header using `python tflite_to_c_header.py`.
6. Upload the firmware to the ESP32 using Arduino IDE.
7. Insert a FAT32-formatted MicroSD card, connect the 24V pump rail, and power the vest.
8. Pair the vest over BLE (device name `NeuroGuard-XXXX`) with the mobile client.
9. Allow the system to complete baseline calibration by wearing the vest still for ~5 seconds.
10. Wear the vest during daily activity - the system continuously monitors neurological state and physically intervenes during emergencies.

To analyze recorded sessions, run `python analyze_neuroguard.py neuroguard_log.csv`. To review multi-session progress and event history, run `python track_neurological_events.py neuroguard_log.csv`.
