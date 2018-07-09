

(define (problem BW-rand-10)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10  - block)
(:init
(handempty)
(on b1 b2)
(ontable b2)
(on b3 b8)
(on b4 b1)
(on b5 b10)
(on b6 b9)
(on b7 b6)
(on b8 b4)
(ontable b9)
(on b10 b3)
(clear b5)
(clear b7)
)
(:goal
(and
(on b1 b5)
(on b2 b8)
(on b3 b9)
(on b4 b2)
(on b5 b3)
(on b6 b10)
(on b7 b1)
(on b9 b4))
)
)


