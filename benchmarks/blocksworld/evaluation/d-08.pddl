

(define (problem BW-rand-8)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8  - block)
(:init
(handempty)
(on b1 b8)
(ontable b2)
(ontable b3)
(on b4 b2)
(on b5 b1)
(on b6 b4)
(ontable b7)
(on b8 b3)
(clear b5)
(clear b6)
(clear b7)
)
(:goal
(and
(on b3 b6)
(on b4 b2)
(on b5 b3)
(on b7 b4))
)
)


