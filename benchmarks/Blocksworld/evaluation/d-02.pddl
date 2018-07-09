

(define (problem BW-rand-2)
(:domain blocksworld)
(:objects b1 b2  - block)
(:init
(handempty)
(ontable b1)
(ontable b2)
(clear b1)
(clear b2)
)
(:goal
(and
(on b2 b1))
)
)


