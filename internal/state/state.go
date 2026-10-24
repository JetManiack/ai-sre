// Package state manages persistent state: fingerprints, quiet windows,
// and long-term investigation memory.
package state

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/glebarez/sqlite"
)

// DB is the state store backed by SQLite.
type DB struct {
	db *sql.DB
}

// Open opens (or creates) the SQLite database at path.
func Open(path string) (*DB, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("state: open %s: %w", path, err)
	}
	db.SetMaxOpenConns(1) // SQLite is single-writer
	s := &DB{db: db}
	if err := s.migrate(); err != nil {
		return nil, err
	}
	return s, nil
}

// Close closes the underlying database.
func (s *DB) Close() error { return s.db.Close() }

func (s *DB) migrate() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS quiet_windows (
			fingerprint TEXT PRIMARY KEY,
			until       INTEGER NOT NULL
		);
		CREATE TABLE IF NOT EXISTS investigations (
			id             TEXT PRIMARY KEY,
			workload       TEXT NOT NULL,
			namespace      TEXT NOT NULL,
			candidate_kind TEXT NOT NULL,
			root_cause_ru  TEXT NOT NULL,
			action         TEXT NOT NULL,
			confidence     REAL NOT NULL,
			occurred_at    INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS investigations_workload
			ON investigations(workload, namespace);
	`)
	return err
}

// IsQuiet returns true if fingerprint is currently in a quiet window.
func (s *DB) IsQuiet(fingerprint string) (bool, error) {
	var until int64
	err := s.db.QueryRow(`SELECT until FROM quiet_windows WHERE fingerprint = ?`, fingerprint).Scan(&until)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return time.Now().Before(time.Unix(until, 0)), nil
}

// ClaimQuiet atomically checks whether fingerprint is quiet and, if not, sets
// it quiet until the given time. Returns true when this caller won the claim
// (i.e. the fingerprint was NOT already quiet). Uses a single SQL statement to
// avoid TOCTOU between concurrent workers.
func (s *DB) ClaimQuiet(fingerprint string, until time.Time) (claimed bool, err error) {
	now := time.Now().Unix()
	res, err := s.db.Exec(`
		INSERT INTO quiet_windows(fingerprint, until) VALUES(?, ?)
		ON CONFLICT(fingerprint) DO UPDATE SET until = excluded.until
		WHERE quiet_windows.until < ?`,
		fingerprint, until.Unix(), now)
	if err != nil {
		return false, err
	}
	n, err := res.RowsAffected()
	return n > 0, err
}

// SetQuiet sets a quiet window for fingerprint until the given time.
// Passing zero time clears the quiet window.
func (s *DB) SetQuiet(fingerprint string, until time.Time) error {
	if until.IsZero() {
		_, err := s.db.Exec(`DELETE FROM quiet_windows WHERE fingerprint = ?`, fingerprint)
		return err
	}
	_, err := s.db.Exec(`
		INSERT INTO quiet_windows(fingerprint, until) VALUES(?, ?)
		ON CONFLICT(fingerprint) DO UPDATE SET until = excluded.until`,
		fingerprint, until.Unix())
	return err
}

// SaveInvestigation persists an investigation record.
func (s *DB) SaveInvestigation(id, workload, namespace, candidateKind, rootCauseRU, action string, confidence float64, occurredAt time.Time) error {
	_, err := s.db.Exec(`
		INSERT OR REPLACE INTO investigations
			(id, workload, namespace, candidate_kind, root_cause_ru, action, confidence, occurred_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		id, workload, namespace, candidateKind, rootCauseRU, action, confidence, occurredAt.Unix())
	return err
}

// RecallInvestigations returns the most recent investigations for a workload.
func (s *DB) RecallInvestigations(workload, namespace string, limit int) ([]Investigation, error) {
	if limit <= 0 {
		limit = 5
	}
	rows, err := s.db.Query(`
		SELECT id, workload, namespace, candidate_kind, root_cause_ru, action, confidence, occurred_at
		FROM investigations
		WHERE workload = ? AND namespace = ?
		ORDER BY occurred_at DESC
		LIMIT ?`, workload, namespace, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Investigation
	for rows.Next() {
		var inv Investigation
		var ts int64
		if err := rows.Scan(&inv.ID, &inv.Workload, &inv.Namespace, &inv.CandidateKind,
			&inv.RootCauseRU, &inv.Action, &inv.Confidence, &ts); err != nil {
			return nil, err
		}
		inv.OccurredAt = time.Unix(ts, 0)
		out = append(out, inv)
	}
	return out, rows.Err()
}

// Investigation is a persisted investigation record.
type Investigation struct {
	ID            string
	Workload      string
	Namespace     string
	CandidateKind string
	RootCauseRU   string
	Action        string
	Confidence    float64
	OccurredAt    time.Time
}
