(define (problem turnandopen-3-10-28)
(:domain turnandopen-strips)
(:objects robot1 robot2 robot3 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 - room
door1 door2 door3 door4 door5 door6 door7 door8 door9 - door
ball1 ball2 ball3 ball4 ball5 ball6 ball7 ball8 ball9 ball10 ball11 ball12 ball13 ball14 ball15 ball16 ball17 ball18 ball19 ball20 ball21 ball22 ball23 ball24 ball25 ball26 ball27 ball28 - object)
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
(at-robby robot1 room1)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room5)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room5)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at ball1 room8)
(at ball2 room1)
(at ball3 room7)
(at ball4 room7)
(at ball5 room2)
(at ball6 room7)
(at ball7 room6)
(at ball8 room4)
(at ball9 room1)
(at ball10 room2)
(at ball11 room2)
(at ball12 room3)
(at ball13 room9)
(at ball14 room8)
(at ball15 room6)
(at ball16 room5)
(at ball17 room5)
(at ball18 room9)
(at ball19 room5)
(at ball20 room10)
(at ball21 room5)
(at ball22 room6)
(at ball23 room7)
(at ball24 room2)
(at ball25 room5)
(at ball26 room5)
(at ball27 room9)
(at ball28 room6)
)
(:goal
(and
(at ball1 room5)
(at ball2 room4)
(at ball3 room10)
(at ball4 room3)
(at ball5 room4)
(at ball6 room6)
(at ball7 room9)
(at ball8 room5)
(at ball9 room3)
(at ball10 room5)
(at ball11 room9)
(at ball12 room3)
(at ball13 room7)
(at ball14 room10)
(at ball15 room6)
(at ball16 room6)
(at ball17 room7)
(at ball18 room1)
(at ball19 room10)
(at ball20 room2)
(at ball21 room10)
(at ball22 room5)
(at ball23 room1)
(at ball24 room4)
(at ball25 room1)
(at ball26 room8)
(at ball27 room6)
(at ball28 room5)
)
)
(:metric minimize (total-cost))

)


