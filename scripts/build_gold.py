"""Resolve hand-written questions to section keys of the pinned FastAPI docs and write data/gold.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_rag.chunking import load_corpus

QUESTIONS: list[tuple[str, str, str]] = [
    ("How can I make a query parameter optional without giving it a default value?", "tutorial/query-params.md", "Optional parameters"),
    ("What does the API return if a path parameter typed as int receives the string 'foo'?", "tutorial/path-params.md", "Data validation"),
    ("How do I restrict a path parameter to a fixed set of allowed values?", "tutorial/path-params.md", "Predefined values"),
    ("How do I capture a URL segment that itself contains slashes, like a file path?", "tutorial/path-params.md", "Path parameters containing paths"),
    ("Why does /users/me need to be declared before /users/{user_id}?", "tutorial/path-params.md", "Order matters"),
    ("How do I limit the maximum length of a string query parameter?", "tutorial/query-params-str-validations.md", "Additional validation"),
    ("How can one query parameter receive several values, like ?q=foo&q=bar?", "tutorial/query-params-str-validations.md", "Query parameter list / multiple values"),
    ("How do I show a parameter as deprecated in the generated docs without removing it?", "tutorial/query-params-str-validations.md", "Deprecating parameters"),
    ("How do I validate that a numeric path value is above zero and at most 1000?", "tutorial/path-params-numeric-validations.md", "Number validations: greater than and less than or equal"),
    ("How do I return a 404 with a custom error message when an item does not exist?", "tutorial/handling-errors.md", "Use `HTTPException`"),
    ("Can I attach custom headers to an error response raised from my endpoint?", "tutorial/handling-errors.md", "Add custom headers"),
    ("How do I change the body returned when the request data fails validation?", "tutorial/handling-errors.md", "Override request validation exceptions"),
    ("How do I make sure a user's password is never sent back in the response?", "tutorial/response-model.md", "Add an output model"),
    ("How can I leave out fields that were never set when returning a model?", "tutorial/response-model.md", "Use the `response_model_exclude_unset` parameter"),
    ("How do I declare that an endpoint answers with HTTP 201 Created?", "tutorial/response-status-code.md", "Response Status Code"),
    ("How do I receive username and password as form fields instead of JSON?", "tutorial/request-forms.md", "Define `Form` parameters"),
    ("How do I get the original filename and content type of an uploaded file?", "tutorial/request-files.md", "`UploadFile`"),
    ("My JavaScript frontend on another domain cannot call my API. How do I allow it?", "tutorial/cors.md", "Use `CORSMiddleware`"),
    ("In CORS terms, what exactly counts as an origin?", "tutorial/cors.md", "Origin"),
    ("How do I send a notification email after the response has already been returned?", "tutorial/background-tasks.md", "Using `BackgroundTasks`"),
    ("How do I write a middleware that adds the processing time as a response header?", "tutorial/middleware.md", "Before and after the `response`"),
    ("How do I split my endpoints across several modules in a large project?", "tutorial/bigger-applications.md", "`APIRouter`"),
    ("How do I give every route of a router the same prefix, tags and dependencies?", "tutorial/bigger-applications.md", "Include an `APIRouter` with a custom `prefix`, `tags`, `responses`, and `dependencies`"),
    ("How do I write tests for my endpoints with pytest?", "tutorial/testing.md", "Using `TestClient`"),
    ("How do I swap a real dependency for a fake one while testing?", "advanced/testing-dependencies.md", "Use the `app.dependency_overrides` attribute"),
    ("How can I test a websocket endpoint?", "advanced/testing-websockets.md", "Testing WebSockets"),
    ("How do I close a database session after the response is sent, from inside a dependency?", "tutorial/dependencies/dependencies-with-yield.md", "A database dependency with `yield`"),
    ("Can a Python class be used as a dependency instead of a function?", "tutorial/dependencies/classes-as-dependencies.md", "Classes as dependencies"),
    ("How do I apply one dependency to every endpoint of the whole application?", "tutorial/dependencies/global-dependencies.md", "Global Dependencies"),
    ("How do I run a dependency whose return value my endpoint does not need?", "tutorial/dependencies/dependencies-in-path-operation-decorators.md", "Add `dependencies` to the *path operation decorator*"),
    ("If two sub-dependencies need the same dependency, is it executed twice per request?", "tutorial/dependencies/sub-dependencies.md", "Using the same dependency multiple times"),
    ("How do I implement a login endpoint that returns a bearer token from username and password?", "tutorial/security/simple-oauth2.md", "Return the token"),
    ("Which library and algorithm does the tutorial recommend for hashing passwords?", "tutorial/security/oauth2-jwt.md", "Install `pwdlib`"),
    ("How do I create signed JWT access tokens with an expiration?", "tutorial/security/oauth2-jwt.md", "Handle JWT tokens"),
    ("How do I obtain the logged-in user object inside my endpoint from the token?", "tutorial/security/get-current-user.md", "Inject the current user"),
    ("Should my endpoint functions be declared with async def or plain def?", "async.md", "In a hurry?"),
    ("What happens when I declare a path operation with normal def instead of async def?", "async.md", "Path operation functions"),
    ("How do I load configuration from environment variables with Pydantic?", "advanced/settings.md", "Pydantic `Settings`"),
    ("How can I avoid re-reading the .env file on every request?", "advanced/settings.md", "Creating the `Settings` only once with `lru_cache`"),
    ("How do I load a machine learning model once at startup and release it at shutdown?", "advanced/events.md", "Lifespan"),
    ("How do I return an HTML page directly from an endpoint?", "advanced/custom-response.md", "`HTMLResponse`"),
    ("How do I stream a large video file to the client?", "advanced/custom-response.md", "`StreamingResponse`"),
    ("How do I redirect the client to a different URL?", "advanced/custom-response.md", "`RedirectResponse`"),
    ("How do I set a cookie on the response?", "advanced/response-cookies.md", "Use a `Response` parameter"),
    ("How can I add custom headers to a successful response?", "advanced/response-headers.md", "Use a `Response` parameter"),
    ("How do I return 201 when an item was created but 200 when it already existed?", "advanced/response-change-status-code.md", "Use a `Response` parameter"),
    ("Besides the main status code, how can an endpoint return other status codes with JSON?", "advanced/additional-status-codes.md", "Additional status codes"),
    ("Can I mount an existing Flask application inside a FastAPI app?", "advanced/wsgi.md", "Using `WSGIMiddleware`"),
    ("How do I mount a separate app under /subapi with its own docs UI?", "advanced/sub-applications.md", "Mounting a **FastAPI** application"),
    ("My app sits behind a proxy that strips an /api/v1 prefix and the docs fail to load. What do I configure?", "advanced/behind-a-proxy.md", "Proxy with a stripped path prefix"),
    ("How do I render pages with Jinja2 templates?", "advanced/templates.md", "Using `Jinja2Templates`"),
    ("How do I create a websocket endpoint?", "advanced/websockets.md", "Create a `websocket`"),
    ("How do I serve images and CSS files from a folder?", "tutorial/static-files.md", "Use `StaticFiles`"),
    ("What should a Dockerfile for a FastAPI app look like?", "deployment/docker.md", "Dockerfile"),
    ("How do I run several worker processes to use all CPU cores?", "deployment/server-workers.md", "Multiple Workers"),
    ("Why should I pin the FastAPI version in my requirements?", "deployment/versions.md", "Pin your `fastapi` version"),
    ("How can one server on a single IP address serve HTTPS certificates for several domains?", "deployment/https.md", "TLS with SNI Extension"),
    ("How do I start the server manually for production without the dev reloader?", "deployment/manually.md", "Use the `fastapi run` Command"),
    ("How can I disable the OpenAPI docs in production through a setting?", "how-to/conditional-openapi.md", "Conditional OpenAPI from settings and env vars"),
    ("How do I change the syntax highlighting theme of Swagger UI?", "how-to/configure-swagger-ui.md", "Change the Theme"),
    ("How do I set the title, description and version shown in the API docs?", "tutorial/metadata.md", "Metadata for API"),
    ("How do I add example values to the JSON schema of a request model?", "tutorial/schema-extra-example.md", "Extra JSON Schema data in Pydantic models"),
    ("How do I convert a model containing datetime into something json.dumps accepts?", "tutorial/encoder.md", "Using the `jsonable_encoder`"),
    ("How do I update only some fields of an item with PATCH?", "tutorial/body-updates.md", "Partial updates with `PATCH`"),
    ("Can parameters be typed as UUID or datetime?", "tutorial/extra-data-types.md", "Other data types"),
    ("Why does a header declared as user_agent match the User-Agent header?", "tutorial/header-params.md", "Automatic conversion"),
    ("How do I store data in a SQL database using SQLModel?", "tutorial/sql-databases.md", "Create Models"),
    ("How do I make every endpoint use a different response class by default?", "advanced/custom-response.md", "Default response class"),
    ("How do I generate a TypeScript client SDK from my API's OpenAPI schema?", "advanced/generate-clients.md", "Create a TypeScript SDK"),
]


def resolve(root: Path) -> list[dict[str, object]]:
    chunks = load_corpus(root)
    titles: dict[str, set[str]] = {}
    for chunk in chunks:
        titles.setdefault(chunk.file, set()).add(f"{chunk.anchor}\t{chunk.heading_path[-1]}")
    rows: list[dict[str, object]] = []
    problems: list[str] = []
    for i, (question, file, heading) in enumerate(QUESTIONS):
        candidates = sorted(
            entry.split("\t")[0] for entry in titles.get(file, set()) if entry.split("\t")[1] == heading
        )
        if len(candidates) != 1:
            problems.append(f"{file} :: {heading!r} -> {candidates}")
            continue
        rows.append({"id": f"q{i + 1:03d}", "question": question, "relevant": [f"{file}#{candidates[0]}"]})
    if problems:
        raise SystemExit("Unresolved gold entries:\n" + "\n".join(problems))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/gold.jsonl"))
    args = parser.parse_args()
    rows = resolve(args.docs)
    args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} questions to {args.out}")


if __name__ == "__main__":
    main()
