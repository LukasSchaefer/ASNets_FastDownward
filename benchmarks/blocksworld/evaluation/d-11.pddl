

(define (problem BW-rand-11)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11  - block)
(:init
(handempty)
(on b1 b2)
(ontable b2)
(on b3 b6)
(on b4 b3)
(on b5 b4)
(on b6 b1)
(ontable b7)
(on b8 b7)
(on b9 b10)
(ontable b10)
(on b11 b8)
(clear b5)
(clear b9)
(clear b11)
)
(:goal
(and
(on b1 b7)
(on b2 b9)
(on b3 b5)
(on b4 b2)
(on b6 b10)
(on b7 b4)
(on b8 b11)
(on b9 b8)
(on b10 b1)
(on b11 b3))
)
)


