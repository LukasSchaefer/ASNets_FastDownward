(define (problem turnandopen-4-13-58)
(:domain turnandopen-strips)
(:objects robot1 robot2 robot3 robot4 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 room11 room12 room13 - room
door1 door2 door3 door4 door5 door6 door7 door8 door9 door10 door11 door12 - door
ball1 ball2 ball3 ball4 ball5 ball6 ball7 ball8 ball9 ball10 ball11 ball12 ball13 ball14 ball15 ball16 ball17 ball18 ball19 ball20 ball21 ball22 ball23 ball24 ball25 ball26 ball27 ball28 ball29 ball30 ball31 ball32 ball33 ball34 ball35 ball36 ball37 ball38 ball39 ball40 ball41 ball42 ball43 ball44 ball45 ball46 ball47 ball48 ball49 ball50 ball51 ball52 ball53 ball54 ball55 ball56 ball57 ball58 - object)
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
(at-robby robot1 room2)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room1)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room11)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room11)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at ball1 room8)
(at ball2 room6)
(at ball3 room7)
(at ball4 room1)
(at ball5 room11)
(at ball6 room12)
(at ball7 room11)
(at ball8 room4)
(at ball9 room2)
(at ball10 room4)
(at ball11 room1)
(at ball12 room10)
(at ball13 room9)
(at ball14 room11)
(at ball15 room7)
(at ball16 room3)
(at ball17 room1)
(at ball18 room10)
(at ball19 room7)
(at ball20 room12)
(at ball21 room10)
(at ball22 room3)
(at ball23 room8)
(at ball24 room8)
(at ball25 room5)
(at ball26 room3)
(at ball27 room10)
(at ball28 room7)
(at ball29 room3)
(at ball30 room8)
(at ball31 room4)
(at ball32 room11)
(at ball33 room13)
(at ball34 room10)
(at ball35 room12)
(at ball36 room11)
(at ball37 room9)
(at ball38 room10)
(at ball39 room1)
(at ball40 room11)
(at ball41 room1)
(at ball42 room2)
(at ball43 room7)
(at ball44 room9)
(at ball45 room13)
(at ball46 room1)
(at ball47 room12)
(at ball48 room13)
(at ball49 room11)
(at ball50 room5)
(at ball51 room11)
(at ball52 room7)
(at ball53 room7)
(at ball54 room6)
(at ball55 room2)
(at ball56 room12)
(at ball57 room8)
(at ball58 room12)
)
(:goal
(and
(at ball1 room6)
(at ball2 room11)
(at ball3 room6)
(at ball4 room10)
(at ball5 room9)
(at ball6 room6)
(at ball7 room7)
(at ball8 room7)
(at ball9 room3)
(at ball10 room3)
(at ball11 room3)
(at ball12 room4)
(at ball13 room13)
(at ball14 room3)
(at ball15 room5)
(at ball16 room7)
(at ball17 room12)
(at ball18 room4)
(at ball19 room7)
(at ball20 room11)
(at ball21 room3)
(at ball22 room5)
(at ball23 room3)
(at ball24 room1)
(at ball25 room11)
(at ball26 room10)
(at ball27 room6)
(at ball28 room13)
(at ball29 room9)
(at ball30 room1)
(at ball31 room11)
(at ball32 room1)
(at ball33 room11)
(at ball34 room4)
(at ball35 room11)
(at ball36 room6)
(at ball37 room9)
(at ball38 room5)
(at ball39 room13)
(at ball40 room11)
(at ball41 room7)
(at ball42 room3)
(at ball43 room2)
(at ball44 room7)
(at ball45 room6)
(at ball46 room6)
(at ball47 room13)
(at ball48 room5)
(at ball49 room9)
(at ball50 room7)
(at ball51 room2)
(at ball52 room12)
(at ball53 room11)
(at ball54 room4)
(at ball55 room12)
(at ball56 room9)
(at ball57 room1)
(at ball58 room4)
)
)
(:metric minimize (total-cost))

)
