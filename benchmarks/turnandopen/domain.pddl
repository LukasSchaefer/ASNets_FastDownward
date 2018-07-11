;; changed from temporal to action cost!

(define (domain turnandopen-strips)
 (:requirements :strips :typing :action-costs) 
 (:types room object robot gripper door) 
 (:predicates (at-robby ?r - robot ?x - room)
 	      (at ?o - object ?x - room)
	      (free ?r - robot ?g - gripper)
	      (carry ?r - robot ?o - object ?g - gripper)
	      (connected ?x - room ?y - room ?d - door)
	      (open ?d - door)
	      (closed ?d - door)
	      (doorknob-turned ?d - door ?g - gripper))

(:functions (total-cost))

   (:action turn-doorknob
       :parameters (?r - robot ?from ?to - room ?d - door ?g - gripper)
       :condition  (and  (over all (at-robby ?r ?from))
       		      	 (at start (free ?r ?g))
			 (over all (connected ?from ?to ?d))
			 (at start (closed ?d)))
       :effect (and  
		    (at start (not (free ?r ?g)))
		    (at end (free ?r ?g))
		    (at start (doorknob-turned ?d ?g))
		    (at end (not (doorknob-turned ?d ?g)))
            (increase (total-cost) 3)))

   (:action open-door
       :parameters (?r - robot ?from ?to - room ?d - door ?g - gripper)
       :condition  (and  (over all (at-robby ?r ?from))
       		      	 (over all (connected ?from ?to ?d))
			 (over all (doorknob-turned ?d ?g))
			 (at start (closed ?d)))
       :effect (and (at start (not (closed ?d)))
		    (at end (open ?d))
            (increase (total-cost) 2)))
	       	    

   (:action move
       :parameters  (?r - robot ?from ?to - room ?d - door)
       :condition (and  (at start (at-robby ?r ?from))
       		     	(over all (connected ?from ?to ?d))
			(over all (open ?d)))
       :effect (and  (at end (at-robby ?r ?to))
		     (at start (not (at-robby ?r ?from)))
             (increase (total-cost) 1)))

   (:action pick
       :parameters (?r - robot ?obj - object ?room - room ?g - gripper)
       :condition  (and  (at start (at ?obj ?room))
       		   	 (at start (at-robby ?r ?room))
			 (at start (free ?r ?g)))
       :effect (and (at end (carry ?r ?obj ?g))
		    (at start (not (at ?obj ?room))) 
		    (at start (not (free ?r ?g)))
            (increase (total-cost) 1)))

   (:action drop
       :parameters (?r - robot ?obj - object ?room - room ?g - gripper)
       :condition  (and  (at start (carry ?r ?obj ?g))
       		   	 (at start (at-robby ?r ?room)))
       :effect (and (at end (at ?obj ?room))
		    (at end (free ?r ?g))
		    (at start (not (carry ?r ?obj ?g)))
            (increase (total-cost) 1)))
)
