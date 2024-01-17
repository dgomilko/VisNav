from pymavlink import mavutil

class MavConnection:
	def __init__(self):
		self.__connection = mavutil.mavlink_connection('/dev/ttyACM0')
		self.__connection.wait_heartbeat()

	def receive_message(self, msg_type: str) -> dict:
		return self.__connection.recv_match(type=msg_type).to_dict()
	
	def receive_gps_position(self) -> dict:
		m_type = 'GLOBAL_POSITION_INT'
		return self.receive_message(m_type)
	
	def receive_angles(self) -> dict:
		m_type = 'ATTITUDE'
		return self.receive_message(m_type)
