

(define (problem BW-rand-20)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20  - block)
(:init
(handempty)
(on b1 b3)
(on b2 b6)
(on b3 b8)
(on b4 b19)
(on b5 b14)
(ontable b6)
(on b7 b16)
(on b8 b2)
(on b9 b17)
(on b10 b1)
(on b11 b13)
(ontable b12)
(on b13 b9)
(on b14 b10)
(on b15 b20)
(on b16 b5)
(on b17 b7)
(on b18 b15)
(ontable b19)
(on b20 b4)
(clear b11)
(clear b12)
(clear b18)
)
(:goal
(and
(on b1 b6)
(on b2 b8)
(on b3 b4)
(on b5 b10)
(on b6 b14)
(on b7 b15)
(on b8 b13)
(on b9 b18)
(on b10 b19)
(on b11 b20)
(on b12 b16)
(on b13 b9)
(on b14 b17)
(on b16 b11)
(on b17 b12)
(on b19 b2)
(on b20 b5))
)
)


