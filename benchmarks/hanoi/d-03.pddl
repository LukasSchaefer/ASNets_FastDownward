(define (problem hanoi-3)
  (:domain hanoi)
  (:objects peg1 peg2 peg3 d1 d2 d3 )
  (:init 
    (smaller d1 peg1)(smaller d1 peg2)(smaller d1 peg3)
    (smaller d2 peg1)(smaller d2 peg2)(smaller d2 peg3)
    (smaller d3 peg1)(smaller d3 peg2)(smaller d3 peg3)

    (smaller d1 d2)(smaller d1 d3)
    (smaller d2 d3)
    
    (clear p1)(clear p2)(clear d1)
    (disk d1)(disk d2)(disk d3)
    (on d1 d2)(on d2 d3)(on d3 peg3)
  )
  (:goal 
    (and (on d1 d2)(on d2 d3)(on d3 peg1) )
  )
)