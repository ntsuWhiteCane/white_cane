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

unsigned int checksum_1;
unsigned int checksum_2;
void setup() {
  // put your setup code here, to run once:
//  Serial1.begin(115200, SERIAL_8N1);
  Serial.begin(115200); //設定Baud rate
}
//int count = 256;
//int flag=2;
int angle = 0;
void loop() {
// put your main code here, to run repeatedly:

 // A1_16_SetPosition(7, CMD_I_JOG,  100, 1000);
 // A1_16_SetPosition(15, CMD_I_JOG,  100, 1000);//(ID,模式,playtime,位置)
 // delay(1000);
  if (angle <= 180){
    angle ++;
  }
  else{
    angle = 0;

  }
  A1_16_SetPosition(ID,CMD_I_JOG, 50, angle);//(ID,模式,playtime,速度)
  //A1_16_SetSpeed(ID,CMD_I_JOG, 1000, 400);
  delay(100);

}


void A1_16_SetPosition(unsigned char _pID, uint16_t _CMD,  unsigned char _playtime, unsigned int _position) {
  static unsigned int _data[5];
  static int _i = 0;
  float angle = _position / 360.0 * 1024;
  _position = (unsigned int)angle;
  _data[0] = _position & 0xff;//lsb最右
  _data[1] = (_position & 0xff00) >> 8; //msb最左
  _data[2] = 0;         //set:0(position control), 1(speed control), 2(torque off), 3(position servo on)
  _data[3] = _pID;     //A1-16 ID
  _data[4] = _playtime;
  checksum_1 = (0x0c)^_pID ^ _CMD;  //package_size^pID^CMD
  for (_i = 0; _i < 5; _i++) checksum_1 ^= _data[_i];
  checksum_1 &= 0xfe;
  checksum_2 = (~checksum_1) & 0xfe;
  Serial.write(0xff);
  Serial.write(0xff);
  Serial.write(0x0c);        //package size//12 7+5 原本+I JOB
  Serial.write(_pID);
  Serial.write(_CMD);
  Serial.write(checksum_1);
  Serial.write(checksum_2);
  for (_i = 0; _i < 5; _i++) Serial.write(_data[_i]);
}


void A1_16_SetSpeed(unsigned char _pID, uint16_t _CMD,  unsigned char _playtime, unsigned int _speed) {
  static unsigned int _data[5];
  static int _i = 0;
  _data[0] = _speed & 0xff;//lsb最右
  _data[1] = (_speed & 0xff00) >> 8; //msb最左
  _data[2] = 1;         //set:0(position control), 1(speed control), 2(torque off), 3(position servo on)
  _data[3] = _pID;
  _data[4] = _playtime;
  checksum_1 = (0x0c)^_pID ^ _CMD;  //package_size^pID^CMD
  for (_i = 0; _i < 5; _i++) checksum_1 ^= _data[_i];
  checksum_1 &= 0xfe;
  checksum_2 = (~checksum_1) & 0xfe;
  Serial.write(0xff);
  Serial.write(0xff);
  Serial.write(0x0c);        //package size
  Serial.write(_pID);
  Serial.write(_CMD);
  Serial.write(checksum_1);
  Serial.write(checksum_2);
  for (_i = 0; _i < 5; _i++){
    Serial.write(_data[_i]);
  }
}

