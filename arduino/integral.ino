#include "I2Cdev.h"

#include "MPU6050_6Axis_MotionApps20.h"
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for SparkFun breakout and InvenSense evaluation board)
// AD0 high = 0x69
MPU6050 mpu;
//MPU6050 mpu(0x69); // <-- use for AD0 high

#define INTERRUPT_PIN 2  // use pin 2 on Arduino Uno & most boards
#define LED_PIN 5
#define PB_PIN 4
bool blinkState = false;
bool pbState = false;
// MPU control/status vars
bool dmpReady = false;  // set true if DMP init was successful
uint8_t mpuIntStatus;   // holds actual interrupt status byte from MPU
uint8_t devStatus;      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[64]; // FIFO storage buffer

// orientation/motion vars
Quaternion q;           // [w, x, y, z]         quaternion container
VectorInt16 aa;         // [x, y, z]            accel sensor measurements
VectorInt16 aaReal;     // [x, y, z]            gravity-free accel sensor measurements
VectorInt16 aaWorld;    // [x, y, z]            world-frame accel sensor measurements
VectorFloat gravity;    // [x, y, z]            gravity vector
float euler[3];         // [psi, theta, phi]    Euler angle container
float ypr[3];           // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector

// packet structure for InvenSense teapot demo
uint8_t teapotPacket[14] = { '$', 0x02, 0,0, 0,0, 0,0, 0,0, 0x00, 0x00, '\r', '\n' };



// ================================================================
// ===               INTERRUPT DETECTION ROUTINE                ===
// ================================================================

volatile bool mpuInterrupt = false;     // indicates whether MPU interrupt pin has gone high
void dmpDataReady() {
    mpuInterrupt = true;
}

#define CMD_EEP_WRITE                      0x01
#define CMD_ACK_EEP_WRITE                  0x41
#define CMD_EEP_READ                       0x02
#define CMD_ACK_EEP_READ                   0x42
#define CMD_RAM_WRITE                      0x03
#define CMD_ACK_RAM_WRITE                  0x43
#define CMD_RAM_READ                       0x04
#define CMD_ACK_RAM_READ                   0x44
#define CMD_I_JOG                          0x05
#define CMD_ACK_I_JOG                      0x45
#define CMD_S_JOG                          0x06
#define CMD_ACK_S_JOG                      0x46
#define CMD_STAT                           0x07
#define CMD_ACK_STAT                       0x47
#define CMD_ROLLBACK                       0x08
#define CMD_ACK_ROLLBACK                   0x48
#define CMD_REBOOT                         0x09
#define CMD_ACK_REBOOT                     0x49
#define ID 16   //A1-16 ID
//--------------------------------------------------------------------------------------------------------------

void A1_16_SetPosition(unsigned char, uint16_t ,  unsigned char , unsigned int);
void A1_16_SetSpeed(unsigned char, uint16_t,  unsigned char , unsigned int);
unsigned int compute_A1_16_angle(int);
unsigned int checksum_1;
unsigned int checksum_2;
int angle = 0;
long count = 0;
// ================================================================
// ===                      INITIAL SETUP                       ===
// ================================================================

void setup() {
    // join I2C bus (I2Cdev library doesn't do this automatically)
    #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        Wire.begin();
        Wire.setClock(400000); // 400kHz I2C clock. Comment this line if having compilation difficulties
        Wire.setWireTimeout(3000);
    #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
        Fastwire::setup(400, true);
    #endif

    // initialize serial communication
    // (115200 chosen because it is required for Teapot Demo output, but it's
    // really up to you depending on your project)
    Serial.begin(115200);
    while (!Serial); // wait for Leonardo enumeration, others continue immediately

    // NOTE: 8MHz or slower host processors, like the Teensy @ 3.3v or Ardunio
    // Pro Mini running at 3.3v, cannot handle this baud rate reliably due to
    // the baud timing being too misaligned with processor ticks. You must use
    // 38400 or slower in these cases, or use some kind of external separate
    // crystal solution for the UART timer.

    // initialize device
    Serial.println(F("Initializing I2C devices..."));
    mpu.initialize();
    pinMode(INTERRUPT_PIN, INPUT);

    // verify connection
    Serial.println(F("Testing device connections..."));
    Serial.println(mpu.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));

    // load and configure the DMP
    Serial.println(F("Initializing DMP..."));
    devStatus = mpu.dmpInitialize();

    // supply your own gyro offsets here, scaled for min sensitivity
    mpu.setXAccelOffset(-2091);
    mpu.setYAccelOffset(-2030);
    mpu.setZAccelOffset(1314);
    mpu.setXGyroOffset(-36);
    mpu.setYGyroOffset(100);
    mpu.setZGyroOffset(-25);

    // mpu.CalibrateGyro(15);
    // mpu.CalibrateAccel(15);

    // make sure it worked (returns 0 if so)
    if (devStatus == 0) {
        // turn on the DMP, now that it's ready
        Serial.println(F("Enabling DMP..."));
        mpu.setDMPEnabled(true);

        // enable Arduino interrupt detection
        Serial.println(F("Enabling interrupt detection (Arduino external interrupt 0)..."));
        attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
        mpuIntStatus = mpu.getIntStatus();

        // set our DMP Ready flag so the main loop() function knows it's okay to use it
        Serial.println(F("DMP ready! Waiting for first interrupt..."));
        dmpReady = true;

        // get expected DMP packet size for later comparison
        packetSize = mpu.dmpGetFIFOPacketSize();
    } else {
        // ERROR!
        // 1 = initial memory load failed
        // 2 = DMP configuration updates failed
        // (if it's going to break, usually the code will be 1)
        Serial.print(F("DMP Initialization failed (code "));
        Serial.print(devStatus);
        Serial.println(F(")"));
    }

    // configure LED for output
    pinMode(LED_PIN, OUTPUT);
    pinMode(PB_PIN, INPUT);
    digitalWrite(PB_PIN, HIGH);
}


void loop() {
    int release = 1;
    
    // if programming failed, don't try to do anything
    if (!dmpReady) return;

    // wait for MPU interrupt or extra packet(s) available
    while (!mpuInterrupt && fifoCount < packetSize) {
        // other program behavior stuff here
        // .
        // .
        // .
        // if you are really paranoid you can frequently test in between other
        // stuff to see if mpuInterrupt is true, and if so, "break;" from the
        // while() loop to immediately process the MPU data
        // .
        // .
        // .
    }
    // while(!pbState){
      
    //   if (!digitalRead(PB_PIN) && release == 1){
    //     release == 0;
    //     delay(20);
    //   }else if (digitalRead(PB_PIN) && release == 0){
    //     release == 1
    //     pbState = true;
    //     digitalWrite(LED_PIN, HIGH);
    //     delay(20);
    //   }
    // }
    // reset interrupt flag and get INT_STATUS byte
    mpuInterrupt = false;
    mpuIntStatus = mpu.getIntStatus();

    // get current FIFO count
    fifoCount = mpu.getFIFOCount();

    // check for overflow (this should never happen unless our code is too inefficient)
    if ((mpuIntStatus & 0x10) || fifoCount == 1024) {
        // reset so we can continue cleanly
        mpu.resetFIFO();
        Serial.println(F("FIFO overflow!"));

    // otherwise, check for DMP data ready interrupt (this should happen frequently)
    } else if (mpuIntStatus & 0x02) {
        // wait for correct available data length, should be a VERY short wait
        while (fifoCount < packetSize) fifoCount = mpu.getFIFOCount();

        // read a packet from FIFO
        mpu.getFIFOBytes(fifoBuffer, packetSize);

        // track FIFO count here in case there is > 1 packet available
        // (this lets us immediately read more without waiting for an interrupt)
        fifoCount -= packetSize;

        // display quaternion values in InvenSense Teapot demo format:
        teapotPacket[2] = fifoBuffer[0];
        teapotPacket[3] = fifoBuffer[1];
        teapotPacket[4] = fifoBuffer[4];
        teapotPacket[5] = fifoBuffer[5];
        teapotPacket[6] = fifoBuffer[8];
        teapotPacket[7] = fifoBuffer[9];
        teapotPacket[8] = fifoBuffer[12];
        teapotPacket[9] = fifoBuffer[13];
        // gyro values
        teapotPacket[10] = fifoBuffer[16];
        teapotPacket[11] = fifoBuffer[17];
        teapotPacket[12] = fifoBuffer[20];
        teapotPacket[13] = fifoBuffer[21];
        teapotPacket[14] = fifoBuffer[24];
        teapotPacket[15] = fifoBuffer[25];
        // accelerometer values
        teapotPacket[16] = fifoBuffer[28];
        teapotPacket[17] = fifoBuffer[29];
        teapotPacket[18] = fifoBuffer[32];
        teapotPacket[19] = fifoBuffer[33];
        teapotPacket[20] = fifoBuffer[36];
        teapotPacket[21] = fifoBuffer[37];
        //temperature
        int16_t temperature = mpu.getTemperature();
        teapotPacket[22] = temperature >> 8;
        teapotPacket[23] = temperature & 0xFF;
        Serial.write(teapotPacket, 28);
        teapotPacket[25]++; // packetCount, loops at 0xFF on purpose
    }
    if(count <= 10){
      count++;  
    }
    else{
      A1_16_SetPosition(ID,CMD_I_JOG, 50, compute_A1_16_angle(angle));//(ID,mode,playtime,angle)
      delay(30);
    }
}

unsigned int compute_A1_16_angle(int angle){
  if(angle >= 90){
    angle = 90;
  }
  if(angle <= -90){
    angle = -90;
  }
  float source_angle = (180 - angle) / 360.0 * 1024;
  unsigned int return_angle = (unsigned int)source_angle;
  return return_angle;
}
void A1_16_SetPosition(unsigned char _pID, uint16_t _CMD,  unsigned char _playtime, unsigned int _position) {
  static unsigned int _data[5];
  static int _i = 0;
  
  _data[0] = _position & 0xff;//lsb
  _data[1] = (_position >> 8 & 0xff); //msb
  _data[2] = 0;         //set:0(position control), 1(speed control), 2(torque off), 3(position servo on)
  _data[3] = _pID;     //A1-16 ID
  _data[4] = _playtime;
  checksum_1 = (0x0c)^_pID ^ _CMD;  //package_size^pID^CMD
  for (_i = 0; _i < 5; _i++){
    checksum_1 ^= _data[_i];
  }
  checksum_1 &= 0xfe;
  checksum_2 = (~checksum_1) & 0xfe;
  Serial.write(0xff);
  Serial.write(0xff);
  Serial.write(0x0c);        //package size//12 7+5 original+I JOB
  Serial.write(_pID);
  Serial.write(_CMD);
  Serial.write(checksum_1);
  Serial.write(checksum_2);
  for (_i = 0; _i < 5; _i++){
    Serial.write(_data[_i]);
  }
}