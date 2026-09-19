# Architecture

`greenbench.cli` is the entry point. Configuration and protocol documents are declarative and hashed. Data preparation writes verified split artifacts; scheduling writes paired counterbalanced plans; initialization supplies framework-neutral tensors; model adapters transpose those tensors into native layouts. Measurement capability detection selects a valid tier. Typed provenance guards every raw record, which is written immutably. Statistics, green metrics, tables, figures and the post-hoc dashboard consume validated raw records without modifying them.

Framework-specific environments communicate through the same run schema, so dependency conflicts do not alter evidence format. Extension-family labels form a hard analysis boundary around the locked core.
