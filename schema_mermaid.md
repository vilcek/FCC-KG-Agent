# Knowledge Graph Schema (Mermaid)

Below are two Mermaid diagrams:
1. ER-style entity definitions with attributes.
2. Relationship / topology diagram showing directed edges (FROM -> TO as defined in the DDL).

---
## 1. ER Diagram
```mermaid
erDiagram
    Site {
        string id PK
        string name
    }
    Unit {
        string id PK
        string name
        string type
    }
    Equipment {
        string id PK
        string tag
        string type
    }
    Stream {
        string id PK
        string name
        string phase
    }
    Tank {
        string id PK
        string name
        string service
    }
    SamplePoint {
        string id PK
        string name
        string location
    }
    Analyzer {
        string id PK
        string name
        string type
    }
    Instrument {
        string id PK
        string name
        string type
    }
    Tag {
        string id PK
        string name
        string eng_unit
    }
    Analyte {
        string code PK
        string name
        string default_unit
    }
    Method {
        string code PK
        string name
    }
    SpecLimit {
        string id PK
        string limit_type
        double lo
        double hi
        string basis
        date valid_from
        date valid_to
    }
    Sample {
        string id PK
        timestamp draw_time
        timestamp receive_time
        string collector_id
    }
    TestResult {
        string id PK
        double value
        string unit
        string status
        timestamp analyzed_time
        double reporting_limit
        double uncertainty
    }
    WorkOrder {
        string id PK
        string type
        timestamp ts
    }
    Event {
        string id PK
        string type
        timestamp ts_start
        timestamp ts_end
    }

    %% Relationships (cardinalities left generic as many-to-many without explicit constraints)
    Site ||--o{ Unit : "HAS_UNIT"
    Equipment }o--o{ Unit : "PART_OF"
    Unit ||--o{ Stream : "PRODUCES_STREAM"
    Tank }o--o{ Stream : "HOLDS"
    SamplePoint }o--o{ Stream : "FOR_STREAM"

    Analyzer }o--o{ Stream : "MEASURES"
    Analyzer }o--o{ Tag : "HAS_TAG"

    Instrument }o--o{ Method : "USES_METHOD"
    Method }o--o{ Analyte : "MEASURES_ANALYTE"

    SpecLimit }o--o{ Stream : "APPLIES_TO"
    SpecLimit }o--o{ Analyte : "FOR_ANALYTE"

    Sample }o--|| Stream : "SAMPLE_OF"
    Sample }o--o{ SamplePoint : "FROM_POINT"

    TestResult }o--|| Sample : "RESULT_OF"
    TestResult }o--o{ Instrument : "TESTED_BY"
    TestResult }o--o{ Method : "TR_USES_METHOD"

    WorkOrder }o--o{ Analyzer : "CALIBRATES"
    Event }o--o{ Unit : "WITHIN_UNIT"

    Analyte }o--o{ Tag : "SERVED_BY_TAG"
```

---
## 2. Directed Relationship Graph
```mermaid
graph LR
    subgraph Process
        Site
        Unit
        Equipment
        Stream
        Tank
        SamplePoint
    end
    subgraph Lab_and_QA
        Analyzer
        Instrument
        Method
        Analyte
        SpecLimit
        Sample
        TestResult
        WorkOrder
        Event
        Tag
    end

    Site -->|HAS_UNIT| Unit
    Equipment -->|PART_OF| Unit
    Unit -->|PRODUCES_STREAM| Stream
    Tank -->|HOLDS| Stream
    SamplePoint -->|FOR_STREAM| Stream

    Analyzer -->|MEASURES| Stream
    Analyzer -->|HAS_TAG| Tag

    Instrument -->|USES_METHOD| Method
    Method -->|MEASURES_ANALYTE| Analyte

    SpecLimit -->|APPLIES_TO| Stream
    SpecLimit -->|FOR_ANALYTE| Analyte

    Sample -->|SAMPLE_OF| Stream
    Sample -->|FROM_POINT| SamplePoint

    TestResult -->|RESULT_OF| Sample
    TestResult -->|TESTED_BY| Instrument
    TestResult -->|TR_USES_METHOD| Method

    WorkOrder -->|CALIBRATES| Analyzer
    Event -->|WITHIN_UNIT| Unit

    Analyte -->|SERVED_BY_TAG| Tag
```

---
### Notes
- Cardinalities are shown generically; adjust (e.g., ||--o{) if you add explicit constraints later.
- Some semantic assumptions (e.g., many-to-many) are made due to absence of explicit constraint metadata in the DDL.
- Feel free to tweak clustering or styling depending on your presentation needs.

### Next Steps / Optional Enhancements
- Add cardinality annotations if domain rules are known (e.g., a Sample belongs to exactly one Stream).
- Generate automated Mermaid from DDL via a small parser script if schema evolves frequently.
- Produce temporal lineage views (Sample -> TestResult) filtered by time windows.
