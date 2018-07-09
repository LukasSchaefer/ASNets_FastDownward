

(define (problem BW-rand-6)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6  - block)
(:init
(handempty)
(on b1 b6)
(on b2 b1)
(on b3 b5)
(ontable b4)
(on b5 b2)
(on b6 b4)
(clear b3)
)
(:goal
(and
(on b1 b2)
(on b2 b5)
(on b5 b6)
(on b6 b4))
)
)


