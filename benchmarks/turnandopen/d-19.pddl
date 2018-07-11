(define (problem turnandopen-5-14-60)
(:domain turnandopen-strips)
(:objects robot1 robot2 robot3 robot4 robot5 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 rgripper5 lgripper5 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 room11 room12 room13 room14 - room
door1 door2 door3 door4 door5 door6 door7 door8 door9 door10 door11 door12 door13 - door
ball1 ball2 ball3 ball4 ball5 ball6 ball7 ball8 ball9 ball10 ball11 ball12 ball13 ball14 ball15 ball16 ball17 ball18 ball19 ball20 ball21 ball22 ball23 ball24 ball25 ball26 ball27 ball28 ball29 ball30 ball31 ball32 ball33 ball34 ball35 ball36 ball37 ball38 ball39 ball40 ball41 ball42 ball43 ball44 ball45 ball46 ball47 ball48 ball49 ball50 ball51 ball52 ball53 ball54 ball55 ball56 ball57 ball58 ball59 ball60 - object)
(:init
(= (total-cost) 0)
(closed door1)
(closed door2)
(closed door3)
(closed door4)
(closed door5)
(closed door6)
(closed door7)
(closed door8)
(closed door9)
(closed door10)
(closed door11)
(closed door12)
(closed door13)
(connected room1 room2 door1)
(connected room2 room1 door1)
(connected room2 room3 door2)
(connected room3 room2 door2)
(connected room3 room4 door3)
(connected room4 room3 door3)
(connected room4 room5 door4)
(connected room5 room4 door4)
(connected room5 room6 door5)
(connected room6 room5 door5)
(connected room6 room7 door6)
(connected room7 room6 door6)
(connected room7 room8 door7)
(connected room8 room7 door7)
(connected room8 room9 door8)
(connected room9 room8 door8)
(connected room9 room10 door9)
(connected room10 room9 door9)
(connected room10 room11 door10)
(connected room11 room10 door10)
(connected room11 room12 door11)
(connected room12 room11 door11)
(connected room12 room13 door12)
(connected room13 room12 door12)
(connected room13 room14 door13)
(connected room14 room13 door13)
(at-robby robot1 room2)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room6)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room3)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room8)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at-robby robot5 room9)
(free robot5 rgripper5)
(free robot5 lgripper5)
(at ball1 room12)
(at ball2 room7)
(at ball3 room9)
(at ball4 room3)
(at ball5 room12)
(at ball6 room8)
(at ball7 room11)
(at ball8 room8)
(at ball9 room10)
(at ball10 room14)
(at ball11 room5)
(at ball12 room12)
(at ball13 room9)
(at ball14 room8)
(at ball15 room3)
(at ball16 room11)
(at ball17 room8)
(at ball18 room3)
(at ball19 room13)
(at ball20 room1)
(at ball21 room6)
(at ball22 room6)
(at ball23 room12)
(at ball24 room1)
(at ball25 room2)
(at ball26 room10)
(at ball27 room2)
(at ball28 room8)
(at ball29 room13)
(at ball30 room10)
(at ball31 room2)
(at ball32 room11)
(at ball33 room3)
(at ball34 room11)
(at ball35 room14)
(at ball36 room1)
(at ball37 room4)
(at ball38 room10)
(at ball39 room9)
(at ball40 room14)
(at ball41 room9)
(at ball42 room13)
(at ball43 room11)
(at ball44 room4)
(at ball45 room7)
(at ball46 room14)
(at ball47 room14)
(at ball48 room1)
(at ball49 room3)
(at ball50 room12)
(at ball51 room2)
(at ball52 room8)
(at ball53 room3)
(at ball54 room13)
(at ball55 room9)
(at ball56 room5)
(at ball57 room9)
(at ball58 room10)
(at ball59 room12)
(at ball60 room7)
)
(:goal
(and
(at ball1 room6)
(at ball2 room14)
(at ball3 room3)
(at ball4 room8)
(at ball5 room10)
(at ball6 room2)
(at ball7 room9)
(at ball8 room14)
(at ball9 room12)
(at ball10 room3)
(at ball11 room14)
(at ball12 room7)
(at ball13 room1)
(at ball14 room11)
(at ball15 room10)
(at ball16 room8)
(at ball17 room10)
(at ball18 room10)
(at ball19 room8)
(at ball20 room13)
(at ball21 room8)
(at ball22 room9)
(at ball23 room6)
(at ball24 room11)
(at ball25 room8)
(at ball26 room14)
(at ball27 room1)
(at ball28 room2)
(at ball29 room10)
(at ball30 room13)
(at ball31 room8)
(at ball32 room2)
(at ball33 room13)
(at ball34 room11)
(at ball35 room9)
(at ball36 room9)
(at ball37 room12)
(at ball38 room3)
(at ball39 room8)
(at ball40 room10)
(at ball41 room6)
(at ball42 room8)
(at ball43 room2)
(at ball44 room7)
(at ball45 room4)
(at ball46 room12)
(at ball47 room14)
(at ball48 room14)
(at ball49 room7)
(at ball50 room8)
(at ball51 room12)
(at ball52 room14)
(at ball53 room2)
(at ball54 room3)
(at ball55 room10)
(at ball56 room10)
(at ball57 room3)
(at ball58 room11)
(at ball59 room11)
(at ball60 room13)
)
)
(:metric minimize (total-cost))

)
