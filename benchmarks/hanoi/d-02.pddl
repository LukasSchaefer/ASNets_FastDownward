(define (problem hanoi-2)
  (:domain hanoi)
  (:objects peg1 peg2 peg3 d1 d2 )
  (:init 
    (smaller d1 peg1)(smaller d1 peg2)(smaller d1 peg3)
    (smaller d2 peg1)(smaller d2 peg2)(smaller d2 peg3)

    (smaller d1 d2)
    
    (clear p1)(clear p2)(clear d1)
    (disk d1)(disk d2)
    (on d1 d2)(on d2 peg3)
  )
  (:goal 
    (and (on d1 d2)(on d2 peg1) )
  )
)