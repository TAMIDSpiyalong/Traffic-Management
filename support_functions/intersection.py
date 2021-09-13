class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y

def ccw(A,B,C):
	return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def check_cross(checkline, traj_line):
    
    previous_x,previous_y = traj_line[1][0],traj_line[1][1]
    next_x,next_y = traj_line[0][0],traj_line[0][1]
    
    return intersect(Point(previous_x,previous_y), Point(next_x,next_y),Point(checkline[0][0],checkline[0][1]),Point(checkline[1][0],checkline[1][1]))

