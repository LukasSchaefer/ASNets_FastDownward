

(define (problem BW-rand-7)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7  - block)
(:init
(handempty)
(on b1 b4)
(on b2 b7)
(on b3 b1)
(on b4 b6)
(ontable b5)
(ontable b6)
(ontable b7)
(clear b2)
(clear b3)
(clear b5)
)
(:goal
(and
(on b2 b6)
(on b3 b5)
(on b5 b2)
(on b6 b4)
(on b7 b1))
)
)


