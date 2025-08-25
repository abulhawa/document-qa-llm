import argparse

from ui.ingest_client import purge_queue, revoke_job_tasks
from core.job_queue import clear_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear job data or purge queues")
    parser.add_argument("--job", help="Job ID to clear")
    parser.add_argument("--purge", action="store_true", help="Purge broker queue")
    parser.add_argument("--queue", default=None, help="Queue name to purge")
    parser.add_argument(
        "--terminate", action="store_true", help="Terminate running tasks when revoking"
    )
    args = parser.parse_args()

    if args.job:
        n = revoke_job_tasks(args.job, terminate=args.terminate)
        print(f"revoked {n} tasks for job {args.job}")
        counts = clear_job(args.job, terminate=args.terminate)
        for k, v in counts.items():
            print(f"removed {v} from {k}")
    elif args.purge:
        n = purge_queue(args.queue)
        qname = args.queue or "celery"
        print(f"purged {n} messages from queue {qname}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
