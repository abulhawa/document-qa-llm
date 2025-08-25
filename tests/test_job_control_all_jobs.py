from core.job_control import set_state, incr_stat, add_task, all_jobs


def test_all_jobs_lists_ids():
    # populate state, stats and tasks for different jobs
    set_state("job_state", "running")
    incr_stat("job_stats", "registered", 1)
    add_task("job_tasks", "task1")

    jobs = all_jobs()
    assert set(["job_state", "job_stats", "job_tasks"]).issubset(set(jobs))
