// SoapScope CNC stage — firmware stub (Arduino / RP2040 / ESP32)
//
// Implements the newline-delimited JSON protocol from
// `python -m soapscope.cli protocol`. This is a skeleton: it parses commands
// and updates a simulated position so you can validate the host link today,
// then wire in real stepper motion at the TODOs.
//
// No external JSON library required — the protocol is small and fixed, so we
// scan for keys directly. Swap in ArduinoJson if you prefer.

const long BAUD = 115200;

// Stage state, in microns (host converts pixels -> microns).
float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
bool  moving = false;

// ---- TODO: driver setup ---------------------------------------------------
// #include <AccelStepper.h>
// AccelStepper stepX(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);
// AccelStepper stepY(AccelStepper::DRIVER, Y_STEP_PIN, Y_DIR_PIN);
// const float UM_PER_STEP = 0.5f;   // calibrate for your leadscrew/microstep

static float readNum(const String& s, const char* key, float dflt) {
  int i = s.indexOf(key);
  if (i < 0) return dflt;
  i = s.indexOf(':', i);
  if (i < 0) return dflt;
  return s.substring(i + 1).toFloat();
}

static void reply(bool ok, const char* err = nullptr) {
  if (ok) {
    Serial.print("{\"ok\":true,\"x\":");  Serial.print(posX, 3);
    Serial.print(",\"y\":");              Serial.print(posY, 3);
    Serial.print(",\"z\":");              Serial.print(posZ, 3);
    Serial.print(",\"moving\":");         Serial.print(moving ? "true" : "false");
    Serial.println("}");
  } else {
    Serial.print("{\"ok\":false,\"err\":\""); Serial.print(err); Serial.println("\"}");
  }
}

static void moveTo(float x, float y, float feed) {
  // TODO: set target steps = (x - posX)/UM_PER_STEP, ... ; run to completion
  //       under an accel profile, clamped to the travel envelope, honouring
  //       `feed` (um/s) as a max rate. Never lose steps.
  moving = true;
  posX = x; posY = y;
  moving = false;
}

void setup() {
  Serial.begin(BAUD);
  // TODO: stepX.setMaxSpeed(...); pinMode(endstops, INPUT_PULLUP); etc.
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line.indexOf("\"move_abs\"") >= 0) {
    float x = readNum(line, "\"x\"", posX);
    float y = readNum(line, "\"y\"", posY);
    float feed = readNum(line, "\"feed\"", 500.0f);
    moveTo(x, y, feed);
    reply(true);
  } else if (line.indexOf("\"move_rel\"") >= 0) {
    float dx = readNum(line, "\"dx\"", 0.0f);
    float dy = readNum(line, "\"dy\"", 0.0f);
    float feed = readNum(line, "\"feed\"", 500.0f);
    moveTo(posX + dx, posY + dy, feed);
    reply(true);
  } else if (line.indexOf("\"home\"") >= 0) {
    // TODO: seek endstops, then zero.
    posX = posY = posZ = 0.0f;
    reply(true);
  } else if (line.indexOf("\"focus\"") >= 0) {
    posZ += readNum(line, "\"dz\"", 0.0f);   // TODO: move Z stepper
    reply(true);
  } else if (line.indexOf("\"stop\"") >= 0) {
    moving = false;                          // TODO: hard-stop motion
    reply(true);
  } else if (line.indexOf("\"ping\"") >= 0) {
    reply(true);
  } else {
    reply(false, "unknown cmd");
  }
}
