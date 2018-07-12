(define (problem hanoi-11)
  (:domain hanoi)
  (:objects peg1 peg2 peg3 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 )
  (:init 
    (smaller d1 peg1)(smaller d1 peg2)(smaller d1 peg3)
    (smaller d2 peg1)(smaller d2 peg2)(smaller d2 peg3)
    (smaller d3 peg1)(smaller d3 peg2)(smaller d3 peg3)
    (smaller d4 peg1)(smaller d4 peg2)(smaller d4 peg3)
    (smaller d5 peg1)(smaller d5 peg2)(smaller d5 peg3)
    (smaller d6 peg1)(smaller d6 peg2)(smaller d6 peg3)
    (smaller d7 peg1)(smaller d7 peg2)(smaller d7 peg3)
    (smaller d8 peg1)(smaller d8 peg2)(smaller d8 peg3)
    (smaller d9 peg1)(smaller d9 peg2)(smaller d9 peg3)
    (smaller d10 peg1)(smaller d10 peg2)(smaller d10 peg3)
    (smaller d11 peg1)(smaller d11 peg2)(smaller d11 peg3)

    (smaller d1 d2)(smaller d1 d3)(smaller d1 d4)(smaller d1 d5)(smaller d1 d6)(smaller d1 d7)(smaller d1 d8)(smaller d1 d9)(smaller d1 d10)(smaller d1 d11)
    (smaller d2 d3)(smaller d2 d4)(smaller d2 d5)(smaller d2 d6)(smaller d2 d7)(smaller d2 d8)(smaller d2 d9)(smaller d2 d10)(smaller d2 d11)
    (smaller d3 d4)(smaller d3 d5)(smaller d3 d6)(smaller d3 d7)(smaller d3 d8)(smaller d3 d9)(smaller d3 d10)(smaller d3 d11)
    (smaller d4 d5)(smaller d4 d6)(smaller d4 d7)(smaller d4 d8)(smaller d4 d9)(smaller d4 d10)(smaller d4 d11)
    (smaller d5 d6)(smaller d5 d7)(smaller d5 d8)(smaller d5 d9)(smaller d5 d10)(smaller d5 d11)
    (smaller d6 d7)(smaller d6 d8)(smaller d6 d9)(smaller d6 d10)(smaller d6 d11)
    (smaller d7 d8)(smaller d7 d9)(smaller d7 d10)(smaller d7 d11)
    (smaller d8 d9)(smaller d8 d10)(smaller d8 d11)
    (smaller d9 d10)(smaller d9 d11)
    (smaller d10 d11)
    
    (clear p1)(clear p2)(clear d1)
    (disk d1)(disk d2)(disk d3)(disk d4)(disk d5)(disk d6)(disk d7)(disk d8)(disk d9)(disk d10)(disk d11)
    (on d1 d2)(on d2 d3)(on d3 d4)(on d4 d5)(on d5 d6)(on d6 d7)(on d7 d8)(on d8 d9)(on d9 d10)(on d10 d11)(on d11 peg3)
  )
  (:goal 
    (and (on d1 d2)(on d2 d3)(on d3 d4)(on d4 d5)(on d5 d6)(on d6 d7)(on d7 d8)(on d8 d9)(on d9 d10)(on d10 d11)(on d11 peg1) )
  )
)