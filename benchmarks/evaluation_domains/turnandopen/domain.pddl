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
       :precondition  (and  (at-robby ?r ?from)
       		      	    (free ?r ?g)
			    (connected ?from ?to ?d)
			    (closed ?d))
       :effect (and  
		    (not (free ?r ?g))
		    (free ?r ?g)
		    (doorknob-turned ?d ?g)
		    (not (doorknob-turned ?d ?g))
                    (increase (total-cost) 3)))

   (:action open-door
       :parameters (?r - robot ?from ?to - room ?d - door ?g - gripper)
       :precondition  (and  (at-robby ?r ?from)
       		      	    (connected ?from ?to ?d)
			    (doorknob-turned ?d ?g)
			    (closed ?d))
       :effect (and (not (closed ?d))
		    (open ?d)
                    (increase (total-cost) 2)))
	       	    

   (:action move
       :parameters  (?r - robot ?from ?to - room ?d - door)
       :precondition (and  (at-robby ?r ?from)
       		     	(connected ?from ?to ?d)
			(open ?d))
       :effect (and  (at-robby ?r ?to)
		     (not (at-robby ?r ?from))
                     (increase (total-cost) 1)))

   (:action pick
       :parameters (?r - robot ?obj - object ?room - room ?g - gripper)
       :precondition  (and  (at ?obj ?room)
       		   	    (at-robby ?r ?room)
			    (free ?r ?g))
       :effect (and (carry ?r ?obj ?g)
		    (not (at ?obj ?room))
		    (not (free ?r ?g))
                    (increase (total-cost) 1)))

   (:action drop
       :parameters (?r - robot ?obj - object ?room - room ?g - gripper)
       :precondition  (and  (carry ?r ?obj ?g)
       		   	    (at-robby ?r ?room))
       :effect (and (at ?obj ?room)
		    (free ?r ?g)
		    (not (carry ?r ?obj ?g))
                    (increase (total-cost) 1)))
)
