#!/usr/bin/env python3
import asyncio
from src.cli import WorkflowReliabilityCLI
async def main():
    cli = WorkflowReliabilityCLI()
    await cli.run()
if __name__ == '__main__':
    asyncio.run(main())