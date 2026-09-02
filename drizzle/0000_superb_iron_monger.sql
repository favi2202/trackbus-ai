CREATE TABLE `passenger_count_events` (
	`event_id` text PRIMARY KEY NOT NULL,
	`schema_version` text NOT NULL,
	`observed_at` text NOT NULL,
	`received_at` text NOT NULL,
	`source` text NOT NULL,
	`bus_id` text NOT NULL,
	`route_id` text NOT NULL,
	`stop_id` text NOT NULL,
	`door_id` text NOT NULL,
	`boardings` integer NOT NULL,
	`alightings` integer NOT NULL,
	`occupancy` integer NOT NULL,
	`capacity` integer NOT NULL,
	`confidence` real NOT NULL,
	`quality_flags` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `passenger_events_observed_idx` ON `passenger_count_events` (`observed_at`);--> statement-breakpoint
CREATE INDEX `passenger_events_bus_observed_idx` ON `passenger_count_events` (`bus_id`,`observed_at`);--> statement-breakpoint
CREATE INDEX `passenger_events_route_observed_idx` ON `passenger_count_events` (`route_id`,`observed_at`);--> statement-breakpoint
CREATE INDEX `passenger_events_source_observed_idx` ON `passenger_count_events` (`source`,`observed_at`);