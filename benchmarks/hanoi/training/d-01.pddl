(define (problem hanoi-1)
  (:domain hanoi-domain)
  (:objects peg1 peg2 peg3 d1 )
  (:init 
    (smaller d1 peg1)(smaller d1 peg2)(smaller d1 peg3)

    
    (clear p1)(clear p2)(clear d1)
    (disk d1)
    (on d1 peg3)
  )
  (:goal 
    (and (on d1 peg1) )
  )
)