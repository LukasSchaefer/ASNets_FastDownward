;; This domain is an adaption of the Monster domain which Sam Toyer used
;; in his MSc thesis. This is a classical planning version of the problem-
;; There is a start location, a goal location and 2 one-way paths of length n
;; to reach the goal. One of the paths has a monster at depth n whichh kills
;; the agent.

(define (domain classical-monster)
	(:requirements :typing :strips :conditional-effects)
    (:types location - object)

    (:constants start finish left-end right-end - location)

    (:predicates (robot-at ?l - location) (has-monster ?l - location)
               (conn ?from ?to - location) (initialised))


    ;;initialised the domain and sets the monster to end of the left hand path
    (:action init-monster-left
           :parameters ()
           :precondition (and (not (initialised)))
           :effect (and (initialised)
                        (has-monster left-end)
            )
    )

    ;;initialised the domain and sets the monster to end of the left hand path
    (:action init-monster-right
           :parameters ()
           :precondition (and (not (initialised)))
           :effect (and (initialised)
                        (has-monster right-end)
            )
    )

    ;; performs an drive action between located actions
    ;; remember the ways are only one-way and only locations at one path 
    ;; are connected. 
    ;; if the drive action drives to a location where the robot is the
    ;; destination is not reached and it leads to a dead-end
    (:action drive
           :parameters (?from ?to - location)
           :precondition (and 	(conn ?from ?to)
           						(robot-at ?from)
           						(initialised)
           				)
           	:effect (and 	(not (robot-at ?from))
           					(when (and (not (has-monster ?from)))
           						(robot-at ?to))
           			)
    )


)