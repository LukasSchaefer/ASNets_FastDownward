

(define (problem BW-rand-9)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9  - block)
(:init
(handempty)
(on b1 b7)
(on b2 b4)
(ontable b3)
(on b4 b9)
(on b5 b1)
(ontable b6)
(on b7 b3)
(on b8 b5)
(on b9 b8)
(clear b2)
(clear b6)
)
(:goal
(and
(on b1 b8)
(on b2 b3)
(on b3 b5)
(on b4 b6)
(on b6 b7)
(on b7 b1))
)
)


